import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time

import numpy as np
import rospy
import tf
import yaml
from PIL import Image
from nav_msgs.msg import OccupancyGrid
from precision_navigation import MapMatcher, external_points, normalize_scan

from localization_quality import scan_quality


OPERATION_LOCK = threading.RLock()

HEADING_FIELDS = ('x', 'y', 'yaw', 'yaw_deg', 'yaw_rad', 'capture_time', 'capture_mode', 'sample_count',
                  'position_spread_mm', 'yaw_spread_deg', 'scan_base', 'laser_extrinsic', 'capture_quality',
                  'validated', 'note')


def heading_of(point):
    return {key: point[key] for key in HEADING_FIELDS if key in point}


def point_headings(point):
    headings = point.get('headings')
    return headings if headings else [heading_of(point)]


def rebuild_point(point, headings):
    primary = headings[0]
    result = dict(point)
    result.update({key: primary[key] for key in HEADING_FIELDS if key in primary})
    result['headings'] = headings
    result['x'] = float(np.mean([heading['x'] for heading in headings]))
    result['y'] = float(np.mean([heading['y'] for heading in headings]))
    result['yaw'] = primary['yaw_deg']
    return result


def write_points(points_file, details_file, points, session=None):
    coordinates = {point['name']: [point['x'], point['y']] for point in points}
    with open(points_file + '.new', 'w', encoding='utf-8') as target:
        yaml.safe_dump(coordinates, target, allow_unicode=True, sort_keys=False)
    with open(details_file + '.new', 'w', encoding='utf-8') as target:
        json.dump({'frame_id': 'map', 'session': session, 'points': points}, target,
                  ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(details_file + '.new', details_file)
    os.replace(points_file + '.new', points_file)


def grid_quality(grid, point):
    cells = np.asarray(grid.data).reshape(grid.info.height, grid.info.width)
    origin = grid.info.origin
    q = origin.orientation
    angle = tf.transformations.euler_from_quaternion((q.x, q.y, q.z, q.w))[2]
    samples = external_points(np.asarray(point['scan_base'], dtype=float))
    yaw = point['yaw_rad']
    rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    world = samples @ rotation.T + np.array([point['x'], point['y']])
    world -= np.array([origin.position.x, origin.position.y])
    local_rotation = np.array([[math.cos(angle), math.sin(angle)], [-math.sin(angle), math.cos(angle)]])
    return scan_quality(cells, grid.info.resolution, world @ local_rotation.T)


def capture_point(monitor, name, mode):
    samples, stamps, scans = [], set(), []
    deadline = time.monotonic() + 6.0
    while len(samples) < 16 and time.monotonic() < deadline:
        monitor.stop()
        with monitor.lock:
            velocity, velocity_stamp = monitor.vel, monitor.vel_stamp
            scan, scan_stamp, odom_stamp = monitor.scan, monitor.scan_stamp, monitor.odom_stamp
        now = time.time()
        if velocity is None or now - velocity_stamp > 0.5 or now - odom_stamp > 0.5:
            raise ValueError('速度或里程计数据更新不足，请检查设备')
        if max(abs(value) for value in velocity) > 0.008:
            raise ValueError('车辆仍在移动，请停稳后记录')
        if scan is None or now - scan_stamp > 0.5:
            raise ValueError('激光数据更新不足，请检查雷达')
        stamp = scan.header.stamp
        key = stamp.to_nsec()
        if key not in stamps:
            monitor.tf_listener.waitForTransform('map', 'base_footprint', stamp, rospy.Duration(1))
            position, orientation = monitor.tf_listener.lookupTransform('map', 'base_footprint', stamp)
            yaw = tf.transformations.euler_from_quaternion(orientation)[2]
            samples.append([position[0], position[1], yaw])
            offset, rotation = monitor.tf_listener.lookupTransform('base_footprint', scan.header.frame_id, stamp)
            laser_yaw = tf.transformations.euler_from_quaternion(rotation)[2]
            ranges = np.asarray(scan.ranges)
            valid = np.isfinite(ranges) & (ranges > max(scan.range_min, 0.15)) & (ranges < min(scan.range_max, 6.0))
            angles = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment + laser_yaw
            scans.append(np.column_stack((offset[0] + ranges[valid] * np.cos(angles[valid]),
                                          offset[1] + ranges[valid] * np.sin(angles[valid]))))
            stamps.add(key)
        rospy.sleep(0.1)
    if len(samples) < 16:
        raise ValueError('连续激光数据不足，点位没有保存')
    values = np.asarray(samples)
    position = np.mean(values[:, :2], axis=0)
    yaw = math.atan2(np.sin(values[:, 2]).mean(), np.cos(values[:, 2]).mean())
    position_span = float(np.max(np.linalg.norm(values[:, :2] - position, axis=1)))
    angle_span = float(np.max(np.abs(np.arctan2(np.sin(values[:, 2] - yaw), np.cos(values[:, 2] - yaw)))))
    if position_span > 0.005 or angle_span > math.radians(0.5):
        raise ValueError('记录期间位置变化 %.1f 毫米、朝向变化 %.2f 度，请等待稳定后记录' %
                         (position_span * 1000, math.degrees(angle_span)))
    point = {'name': name, 'x': float(position[0]), 'y': float(position[1]),
             'yaw': math.degrees(yaw), 'yaw_deg': math.degrees(yaw), 'yaw_rad': yaw,
             'time': time.strftime('%m-%d %H:%M'), 'capture_time': time.strftime('%Y-%m-%dT%H:%M:%S'),
             'capture_mode': mode, 'sample_count': len(samples),
             'position_spread_mm': position_span * 1000, 'yaw_spread_deg': math.degrees(angle_span),
             'scan_base': scans[-1].tolist(), 'validated': False}
    point['laser_extrinsic'] = [float(offset[0]), float(offset[1]), float(laser_yaw)]
    with monitor.lock:
        grid = monitor.map_msg
    if grid is None:
        raise ValueError('当前没有地图，点位没有保存')
    point['capture_quality'] = grid_quality(grid, point)
    point['headings'] = [heading_of(point)]
    return point


def capture_heading(monitor, point, mode, note=''):
    capture = capture_point(monitor, point['name'], mode)
    headings = [dict(heading) for heading in point_headings(point)]
    heading = heading_of(capture)
    heading['note'] = note
    headings.append(heading)
    return rebuild_point(point, headings)


def validate_points(grid, points, geometry):
    if not points:
        raise ValueError('尚未记录点位，请记录当前位置后保存地图')
    verified = []
    cells = np.asarray(grid.data).reshape(grid.info.height, grid.info.width)
    origin = grid.info.origin
    q = origin.orientation
    angle = tf.transformations.euler_from_quaternion((q.x, q.y, q.z, q.w))[2]
    matcher = MapMatcher(cells, grid.info.resolution, (origin.position.x, origin.position.y, angle))
    for point in points:
        checked = []
        for heading in point_headings(point):
            if 'scan_base' not in heading:
                raise ValueError('点位 %s 没有本次建图的激光记录，请重新记录' % point['name'])
            source = {**heading, 'name': point['name'],
                      'recording_session': point.get('recording_session', heading.get('recording_session'))}
            normalized = normalize_scan(source, geometry)
            projection = matcher.project(normalized)
            quality = grid_quality(grid, {**normalized, **projection})
            if not quality['coverage_ok'] or quality['wall_percent'] is None or quality['wall_percent'] < 95 or quality['all_percent'] < 85:
                raise ValueError('点位 %s 朝向 %.1f 度与最终地图检查未通过：墙面 %s%%，全部激光 %.1f%%，两个方向覆盖 %s' %
                                 (point['name'], heading['yaw_deg'], quality['wall_percent'],
                                  quality['all_percent'], quality['coverage_ok']))
            source_extrinsic = heading.get('laser_extrinsic', geometry['legacy_extrinsic'])
            checked.append({**heading, 'laser_extrinsic': source_extrinsic,
                            'map_alignment': projection, 'final_quality': quality, 'validated': True})
        verified.append(rebuild_point(point, checked))
    return verified


def saved_grid(path):
    from pathlib import Path

    path = Path(path)
    metadata = yaml.safe_load(path.read_text(encoding='utf-8'))
    image = np.asarray(Image.open(path.parent / metadata['image']).convert('L'), dtype=float)[::-1]
    probability = image / 255 if metadata['negate'] else (255 - image) / 255
    cells = np.full(image.shape, -1, dtype=np.int8)
    cells[probability > metadata['occupied_thresh']] = 100
    cells[probability < metadata['free_thresh']] = 0
    grid = OccupancyGrid()
    grid.info.height, grid.info.width = cells.shape
    grid.info.resolution = metadata['resolution']
    grid.info.origin.position.x, grid.info.origin.position.y = metadata['origin'][:2]
    grid.info.origin.orientation.z = math.sin(metadata['origin'][2] / 2)
    grid.info.origin.orientation.w = math.cos(metadata['origin'][2] / 2)
    grid.data = cells.ravel().tolist()
    return grid


def pair_manifest(map_dir, session):
    filenames = ('field_map.pgm', 'field_map.yaml', 'taught_points.yaml', 'taught_points.json')
    digests = {}
    for name in filenames:
        with open(os.path.join(map_dir, name), 'rb') as source:
            digests[name] = hashlib.sha256(source.read()).hexdigest()
    return {'session': session, 'saved_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'sha256': digests,
            'precision_note': '稳定性数值仅表示记录期间变化，实际导航精度需要现场测量'}
