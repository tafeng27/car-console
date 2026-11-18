import copy
from functools import lru_cache
import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

from point_navigation import GridPlanner
from precision_navigation import MapMatcher, body_mask, navigation_cells, normalize_scan


def load_grid(path, points=None):
    path = Path(path)
    metadata = yaml.safe_load(path.read_text(encoding="utf-8"))
    image = np.asarray(Image.open(path.parent / metadata["image"]).convert("L"), dtype=float)[::-1]
    probability = image / 255 if metadata["negate"] else (255 - image) / 255
    cells = np.full(image.shape, -1, dtype=np.int16)
    cells[probability > metadata["occupied_thresh"]] = 100
    cells[probability < metadata["free_thresh"]] = 0
    if points is not None:
        cells, removed = navigation_cells(cells, metadata['resolution'], metadata['origin'], points)
    return GridPlanner(cells, metadata["resolution"], metadata["origin"])


@lru_cache(maxsize=2)
def _project_points(metadata_json, image_bytes, points_json, geometry_json):
    metadata = json.loads(metadata_json)
    image = np.asarray(Image.open(io.BytesIO(image_bytes)).convert('L'), dtype=float)[::-1]
    probability = image / 255 if metadata['negate'] else (255 - image) / 255
    cells = np.full(image.shape, -1, dtype=np.int16)
    cells[probability > metadata['occupied_thresh']] = 100
    cells[probability < metadata['free_thresh']] = 0
    matcher = MapMatcher(cells, metadata['resolution'], metadata['origin'])
    points = json.loads(points_json)
    geometry = json.loads(geometry_json)
    for name, source in points.items():
        session = source.get('recording_session')
        prepared = []
        for heading in source.get('headings') or [source]:
            recorded = {**heading, 'name': name,
                        'recording_session': heading.get('recording_session', session)}
            point = normalize_scan(recorded, geometry)
            projected = matcher.project(point)
            samples = np.asarray(point['scan_base'])
            selected = body_mask(samples)
            point['self_returns'] = samples[selected].tolist()
            point['scan_base'] = samples[~selected].tolist()
            point['recorded_pose'] = {key: heading[key] for key in ('x', 'y', 'yaw_deg')}
            point.update(projected)
            point['scan_pose'] = [point['x'], point['y'], math.radians(point['yaw_deg'])]
            prepared.append(point)
        primary = prepared[0]
        entry = {**source, **primary}
        entry['x'] = float(np.mean([heading['x'] for heading in prepared]))
        entry['y'] = float(np.mean([heading['y'] for heading in prepared]))
        entry['yaw_deg'] = primary['yaw_deg']
        entry['yaw'] = primary['yaw_deg']
        entry['yaw_rad'] = primary['yaw_rad']
        entry['scan_pose'] = list(primary['scan_pose'])
        entry['headings'] = prepared
        points[name] = entry
    return points


def prepare_points(config, points, config_path):
    path = (Path(config_path).resolve().parent / config['map']).resolve()
    metadata = yaml.safe_load(path.read_text(encoding='utf-8'))
    projected = _project_points(json.dumps(metadata), (path.parent / metadata['image']).read_bytes(),
                                json.dumps(points, ensure_ascii=False, allow_nan=False), json.dumps(config['laser_geometry']))
    result = copy.deepcopy(projected)
    calibration_path = (Path(config_path).resolve().parent / config['arrival_calibration_file']).resolve()
    calibration = json.loads(calibration_path.read_text(encoding='utf-8'))
    if calibration['version'] != 1:
        raise ValueError('停车校正记录版本不支持')
    sessions = {point['recording_session'] for point in result.values()}
    if len(sessions) != 1:
        raise ValueError('导航点位需要属于同一次地图保存')
    offsets = calibration['sessions'].get(next(iter(sessions)), {})
    for name, offset in offsets.items():
        if name not in result or offset['basis'] not in ('recorded_heading', 'heading_at_confirmation'):
            raise ValueError('停车校正点位或参考方向无效')
        forward, left = offset['forward_m'], offset['left_m']
        if not all(math.isfinite(value) for value in (forward, left)) or math.hypot(forward, left) > 0.10:
            raise ValueError('人工测量的停车校正需要在10厘米以内')
        point = result[name]
        heading = point['yaw_deg'] if offset['basis'] == 'recorded_heading' else offset['heading_deg']
        if not math.isfinite(heading):
            raise ValueError('停车校正的测量朝向无效')
        angle = math.radians(heading)
        point['x'] += math.cos(angle) * forward - math.sin(angle) * left
        point['y'] += math.sin(angle) * forward + math.cos(angle) * left
        point['arrival_calibration'] = offset
    return result


def point_heading(config, points, name):
    return points[name]["yaw_deg"]
