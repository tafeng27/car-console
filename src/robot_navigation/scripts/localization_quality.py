import math

import cv2
import numpy as np


def scan_quality(cells, resolution, local_points):
    occupied = np.asarray(cells) >= 65
    rows, columns = np.nonzero(occupied)
    if not len(rows) or not len(local_points):
        raise ValueError("定位检查需要有效地图和激光数据")
    points = np.asarray(local_points)
    occupied_x, occupied_y = (columns + 0.5) * resolution, (rows + 0.5) * resolution
    errors = np.full(len(points), np.inf)
    inside = ((points[:, 0] >= 0) & (points[:, 0] < cells.shape[1] * resolution)
              & (points[:, 1] >= 0) & (points[:, 1] < cells.shape[0] * resolution))
    indices = np.flatnonzero(inside)
    for begin in range(0, len(indices), 64):
        selection = indices[begin:begin + 64]
        ex = np.maximum(np.abs(points[selection, 0, None] - occupied_x[None]) - resolution / 2, 0)
        ey = np.maximum(np.abs(points[selection, 1, None] - occupied_y[None]) - resolution / 2, 0)
        errors[selection] = np.sqrt(np.min(ex ** 2 + ey ** 2, axis=1))
    # 从地图中提取至少80厘米的连续直线，单独检查墙面。
    lines = cv2.HoughLinesP(occupied.astype(np.uint8) * 255, 1, np.pi / 720,
                            threshold=max(10, round(0.6 / resolution)),
                            minLineLength=round(0.8 / resolution), maxLineGap=round(0.1 / resolution))
    wall = np.zeros(len(points), dtype=bool)
    horizontal, vertical = wall.copy(), wall.copy()
    if lines is not None:
        for raw in np.asarray(lines).reshape(-1, 4):
            first, last = (raw[:2] + 0.5) * resolution, (raw[2:] + 0.5) * resolution
            segment = last - first
            length = float(np.linalg.norm(segment))
            if length < 0.8:
                continue
            direction = segment / length
            projection = np.clip((points - first) @ direction, 0, length)
            distance = np.linalg.norm(points - first - projection[:, None] * direction, axis=1)
            selected = inside & (distance <= 0.18)
            wall |= selected
            if abs(direction[0]) >= math.cos(math.radians(15)):
                horizontal |= selected
            if abs(direction[1]) >= math.cos(math.radians(15)):
                vertical |= selected
    matched = errors <= 0.100001
    coverage = int(wall.sum()) >= 60 and int(horizontal.sum()) >= 20 and int(vertical.sum()) >= 20
    return {"all_percent": float(100 * matched.mean()),
            "wall_percent": float(100 * matched[wall].mean()) if wall.any() else None,
            "all_points": len(points), "wall_points": int(wall.sum()),
            "horizontal_points": int(horizontal.sum()), "vertical_points": int(vertical.sum()),
            "coverage_ok": coverage, "unmatched_wall_points": int((wall & ~matched).sum()),
            "unmatched_other_points": int((~wall & ~matched).sum())}


def require_quality(quality):
    if not quality["coverage_ok"]:
        raise ValueError("墙面检查需要两个方向的有效激光数据")
    if quality["wall_percent"] < 95:
        raise ValueError("墙面贴合率%.1f%%，需要达到95%%" % quality["wall_percent"])
    if quality["all_percent"] < 85:
        raise ValueError("全部激光匹配率%.1f%%，需要达到85%%" % quality["all_percent"])
