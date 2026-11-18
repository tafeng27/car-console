import math
import json
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree


def validate_settings(settings):
    ranges = {'position_tolerance': (0.002, 0.02), 'heading_tolerance_deg': (0.5, 3),
              'stable_position': (0.001, 0.005), 'stable_heading_deg': (0.1, 1),
              'settle_timeout': (2, 10), 'max_estimator_difference': (0.02, 0.10),
              'max_estimator_heading_deg': (2, 12),
              'latch_latency': (0.02, 0.2),
              'max_correction_distance': (0.02, 0.12), 'correction_speed': (0.12, 0.15),
              'turn_speed': (0.4, 0.6), 'motion_timeout': (2, 6),
              'confirm_scans': (3, 10), 'max_corrections': (1, 8)}
    if set(settings) != set(ranges):
        raise ValueError('精确停车参数不完整或包含未知参数')
    for key, (minimum, maximum) in ranges.items():
        if not math.isfinite(settings[key]) or not minimum <= settings[key] <= maximum:
            raise ValueError('精确停车参数%s超过允许范围' % key)
    if any(not isinstance(settings[key], int) for key in ('confirm_scans', 'max_corrections')):
        raise ValueError('激光确认数量与调整次数需要为整数')


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def transform(points, pose):
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return np.asarray(points) @ np.array([[c, s], [-s, c]]) + np.asarray(pose[:2])


def body_mask(points):
    points = np.asarray(points)
    return (np.abs(points[:, 0]) <= 0.167) & (np.abs(points[:, 1]) <= 0.1515)


def external_points(points):
    points = np.asarray(points)
    return points[~body_mask(points)]


def normalize_scan(point, geometry):
    source = point.get('laser_extrinsic')
    if source is None:
        if point.get('recording_session') != geometry['legacy_session']:
            raise ValueError('点位%s缺少记录时的雷达安装参数' % point['name'])
        source = geometry['legacy_extrinsic']
    target = geometry['current_extrinsic']
    angle = target[2] - source[2]
    samples = transform(np.asarray(point['scan_base']) - np.asarray(source[:2]), (target[0], target[1], angle))
    return {**point, 'scan_base': samples.tolist(), 'laser_extrinsic': list(target)}


def navigation_cells(cells, resolution, origin, points):
    result = np.asarray(cells).copy()
    count, labels, statistics, _ = cv2.connectedComponentsWithStats((result >= 65).astype(np.uint8), 8)
    rows, columns = np.nonzero((result >= 65) & (statistics[labels, cv2.CC_STAT_AREA] == 1))
    external, internal = [], []
    for point in points.values():
        pose = tuple(point['scan_pose'])
        external.extend(transform(external_points(point['scan_base']), pose))
        if point.get('self_returns'):
            internal.extend(transform(point['self_returns'], pose))
    if not internal:
        return result, []
    external_tree, internal_tree = cKDTree(external), cKDTree(internal)
    removed = []
    for row, column in zip(rows, columns):
        center = transform([[(column + 0.5) * resolution, (row + 0.5) * resolution]], origin)[0]
        distance_external, _ = external_tree.query(center)
        distance_internal, _ = internal_tree.query(center)
        if distance_external > 0.10 and distance_internal < resolution:
            result[row, column] = 0
            removed.append({'row': int(row), 'column': int(column), 'xy': center.tolist(),
                            'nearest_external_m': float(distance_external), 'nearest_body_return_m': float(distance_internal)})
    return result, removed


def scan_sample(listener, scan):
    import rospy
    import tf

    listener.waitForTransform('map', 'base_footprint', scan.header.stamp, rospy.Duration(0.5))
    position, quaternion = listener.lookupTransform('map', 'base_footprint', scan.header.stamp)
    offset, rotation = listener.lookupTransform('base_footprint', scan.header.frame_id, scan.header.stamp)
    yaw = tf.transformations.euler_from_quaternion(rotation)[2]
    ranges = np.asarray(scan.ranges)
    valid = np.isfinite(ranges) & (ranges > max(scan.range_min, 0.15)) & (ranges < min(scan.range_max, 6))
    angles = scan.angle_min + np.arange(len(ranges))[valid] * scan.angle_increment + yaw
    points = np.column_stack((offset[0] + ranges[valid] * np.cos(angles), offset[1] + ranges[valid] * np.sin(angles)))
    return external_points(points), (position[0], position[1], tf.transformations.euler_from_quaternion(quaternion)[2])


class MapMatcher:
    def __init__(self, cells, resolution, origin):
        occupied = (np.asarray(cells) >= 65).astype(np.uint8)
        if occupied.sum() < 40:
            raise ValueError('地图占用区域不足，无法检查点位')
        factor = max(1, round(resolution / 0.005))
        occupied = cv2.resize(occupied, None, fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)
        self.resolution = resolution / factor
        self.distance = cv2.distanceTransform(1 - occupied, cv2.DIST_L2, cv2.DIST_MASK_PRECISE) * self.resolution
        self.origin = origin

    def errors(self, points, poses):
        poses = np.asarray(poses)
        yaw = poses[:, 2, None]
        x = poses[:, 0, None] + np.cos(yaw) * points[None, :, 0] - np.sin(yaw) * points[None, :, 1] - self.origin[0]
        y = poses[:, 1, None] + np.sin(yaw) * points[None, :, 0] + np.cos(yaw) * points[None, :, 1] - self.origin[1]
        c, s = math.cos(self.origin[2]), math.sin(self.origin[2])
        return cv2.remap(self.distance, ((c * x + s * y) / self.resolution - 0.5).astype(np.float32),
                         ((-s * x + c * y) / self.resolution - 0.5).astype(np.float32), cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=1)

    def project(self, point):
        points = external_points(np.asarray(point['scan_base'], dtype=float))
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 100 or not np.isfinite(points).all():
            raise ValueError('点位%s的记录激光不足' % point['name'])
        original = np.array([point['x'], point['y'], math.radians(point['yaw_deg'])])
        center = original.copy()
        for radius, step, angle, angular_step in ((0.20, 0.02, 5, 0.5), (0.02, 0.005, 0.5, 0.125)):
            offsets = np.array(np.meshgrid(np.arange(-radius, radius + step / 2, step),
                                           np.arange(-radius, radius + step / 2, step),
                                           np.radians(np.arange(-angle, angle + angular_step / 2, angular_step)))).reshape(3, -1).T
            candidates = center + offsets
            scores = []
            for begin in range(0, len(candidates), 256):
                errors = self.errors(points, candidates[begin:begin + 256])
                scores.extend(np.mean(np.minimum(errors, 0.10) ** 2, axis=1))
            center = candidates[int(np.argmin(scores))]
        errors = self.errors(points, center[None])[0]
        if np.mean(errors <= 0.03) < 0.85 or np.mean(np.minimum(errors, 0.1)) > 0.015:
            raise ValueError('点位%s与最终地图不能可靠匹配，请检查记录激光' % point['name'])
        if np.max(np.abs(center[:2] - original[:2])) >= 0.19 or abs(wrap(center[2] - original[2])) >= math.radians(4.8):
            raise ValueError('点位%s的地图匹配达到搜索边界' % point['name'])
        # 对厘米级的其他位置保留歧义检查，地图匹配不用于证明毫米精度。
        alternatives = np.tile(center, (4, 1))
        alternatives[:, :2] += np.array([[0.04, 0], [-0.04, 0], [0, 0.04], [0, -0.04]])
        scores = np.mean(np.minimum(self.errors(points, np.vstack((center, alternatives))), 0.1) ** 2, axis=1)
        if float(np.min(scores[1:]) - scores[0]) < 0.000002:
            raise ValueError('点位%s的地图匹配位置存在歧义' % point['name'])
        return {'x': float(center[0]), 'y': float(center[1]), 'yaw_deg': math.degrees(center[2]),
                'yaw': math.degrees(center[2]), 'yaw_rad': float(center[2]),
                'projection_quality': {'mean_m': float(np.mean(np.minimum(errors, 0.1))),
                                       'within_3cm': float(np.mean(errors <= 0.03)),
                                       'translation_m': float(np.linalg.norm(center[:2] - original[:2]))}}


class ReferenceMatcher:
    """一个点位可以有多个朝向，每个朝向各有一份记录激光，取匹配最好的那份。"""

    def __init__(self, point):
        self.references = [_Reference(heading) for heading in point.get('headings') or [point]]

    def match(self, points, prior):
        best = None
        failures = []
        for reference in self.references:
            try:
                fitted, quality = reference.match(points, prior)
            except RuntimeError as error:
                failures.append(str(error))
                continue
            if best is None or quality['inlier_fraction'] > best[1]['inlier_fraction']:
                best = (fitted, quality)
        if best is None:
            raise RuntimeError(failures[0])
        return best


class _Reference:
    def __init__(self, point):
        self.target = tuple(point['scan_pose'])
        self.points = transform(external_points(point['scan_base']), self.target)
        self.tree = cKDTree(self.points)
        _, indices = self.tree.query(self.points, k=7)
        neighbors = self.points[indices]
        centered = neighbors - neighbors.mean(axis=1, keepdims=True)
        eigenvalues, vectors = np.linalg.eigh(np.einsum('nki,nkj->nij', centered, centered))
        self.normals = vectors[:, :, 0]
        self.valid = (eigenvalues[:, 1] > 0.0002) & (eigenvalues[:, 0] < eigenvalues[:, 1] * 0.08)
        if self.valid.sum() < 60:
            raise ValueError('记录激光缺少稳定直线特征')
        self.all_tree, self.all_valid = self.tree, self.valid
        self.points, self.normals = self.points[self.valid], self.normals[self.valid]
        self.tree = cKDTree(self.points)

    def match(self, points, prior):
        points, prior = external_points(points), np.asarray(prior)
        if len(points) < 100:
            raise RuntimeError('到点检查的激光数据不足')
        distance, indices = self.all_tree.query(transform(points, prior))
        selected = (distance < 0.20) & self.all_valid[indices]
        if selected.sum() < 60:
            raise RuntimeError('当前激光与记录点位的共同特征不足')
        source = points[selected]

        def residual(pose):
            world = transform(source, pose)
            distance, nearest = self.tree.query(world)
            signed = np.einsum('ij,ij->i', world - self.points[nearest], self.normals[nearest])
            return np.where(distance < 0.20, signed, 0.1)

        def jacobian(pose):
            distance, nearest = self.tree.query(transform(source, pose))
            normal = self.normals[nearest]
            c, s = math.cos(pose[2]), math.sin(pose[2])
            angular = np.column_stack((-s * source[:, 0] - c * source[:, 1],
                                        c * source[:, 0] - s * source[:, 1]))
            values = np.column_stack((normal, np.einsum('ij,ij->i', angular, normal)))
            values[distance >= 0.20] = 0
            return values

        bound = np.array([0.15, 0.15, math.radians(8)])
        fitted = least_squares(residual, prior, jac=jacobian, bounds=(prior - bound, prior + bound),
                               loss='soft_l1', f_scale=0.01, x_scale=[0.05, 0.05, 0.05], max_nfev=60)
        errors = np.abs(residual(fitted.x))
        good = errors <= 0.02
        information = fitted.jac[good].T @ fitted.jac[good]
        eigenvalues = np.linalg.eigvalsh(information)
        if not fitted.success or np.mean(good) < 0.75 or eigenvalues[0] < 1 or eigenvalues[-1] / eigenvalues[0] > 1000:
            raise RuntimeError('记录激光匹配未通过：优化状态%s，有效比例%.3f，约束%s，求解次数%d，初始%s，结果%s' %
                               (fitted.status, np.mean(good), eigenvalues.tolist(), fitted.nfev, prior.tolist(), fitted.x.tolist()))
        if np.max(np.abs((fitted.x - prior) / bound)) > 0.9:
            raise RuntimeError('激光位置匹配达到搜索边界')
        return tuple(fitted.x), {'median_m': float(np.median(errors)), 'inlier_fraction': float(np.mean(good)),
                                'condition': float(eigenvalues[-1] / eigenvalues[0])}


class PrecisionStop:
    def __init__(self, motion, settings):
        validate_settings(settings)
        self.motion = motion
        self.settings = settings
        self.references = {}

    def observe(self, goal):
        motion = self.motion
        motion.stop()
        motion.verify_geometry(goal['laser_extrinsic'])
        deadline = time.monotonic() + self.settings['settle_timeout']
        samples, stamps = [], set()
        if goal['name'] not in self.references:
            self.references[goal['name']] = ReferenceMatcher(goal)
        reference = self.references[goal['name']]
        while time.monotonic() < deadline:
            motion.check()
            scan, points, mapped, velocity = motion.sample()
            if scan in stamps:
                time.sleep(0.03)
                continue
            stamps.add(scan)
            if max(abs(velocity[0]), abs(velocity[1])) > 0.005 or abs(velocity[2]) > 0.02:
                samples = []
                continue
            motion.stationary_update()
            fitted, quality = reference.match(points, mapped)
            samples.append((scan, fitted, mapped, quality))
            samples = samples[-self.settings['confirm_scans']:]
            if len(samples) < self.settings['confirm_scans']:
                continue
            poses = np.asarray([value[1] for value in samples])
            mapped_poses = np.asarray([value[2] for value in samples])
            position = np.median(poses[:, :2], axis=0)
            yaw = math.atan2(np.sin(poses[:, 2]).mean(), np.cos(poses[:, 2]).mean())
            mapped_yaw = math.atan2(np.sin(mapped_poses[:, 2]).mean(), np.cos(mapped_poses[:, 2]).mean())
            stable = (np.max(np.linalg.norm(poses[:, :2] - position, axis=1)) <= self.settings['stable_position']
                      and max(abs(wrap(value - yaw)) for value in poses[:, 2]) <= math.radians(self.settings['stable_heading_deg'])
                      and np.max(np.linalg.norm(mapped_poses[:, :2] - np.median(mapped_poses[:, :2], axis=0), axis=1)) <= self.settings['stable_position']
                      and max(abs(wrap(value - mapped_yaw)) for value in mapped_poses[:, 2]) <= math.radians(self.settings['stable_heading_deg']))
            if stable:
                if any(math.dist(value[1][:2], value[2][:2]) > self.settings['max_estimator_difference']
                       or abs(wrap(value[1][2] - value[2][2])) > math.radians(self.settings['max_estimator_heading_deg'])
                       for value in samples):
                    raise RuntimeError('地图位置与记录激光匹配位置差异过大，车辆保持停车')
                result = (float(position[0]), float(position[1]), yaw)
                quality = {**samples[-1][3],
                    'position_spread_m': float(np.max(np.linalg.norm(poses[:, :2] - position, axis=1))),
                    'mapped_pose': list(samples[-1][2])}
                motion.reconcile(goal, result, quality, points)
                return result, quality
        raise RuntimeError('停车后的定位或速度未能连续稳定，车辆保持停车')

    def finish(self, goal, heading, transit=False, transit_tolerance=0.02, stopped_left=None):
        if not 0.005 <= transit_tolerance <= 0.02:
            raise ValueError('路线通行检查范围需要为5至20毫米')
        motion = self.motion
        began = time.monotonic()
        for attempt in range(self.settings['max_corrections'] + 1):
            pose, quality = self.observe(goal)
            error = np.array([goal['x'] - pose[0], goal['y'] - pose[1]])
            distance = float(np.linalg.norm(error))
            angle = wrap(heading - pose[2])
            tolerance = transit_tolerance if transit else self.settings['position_tolerance']
            evidence = {'goal': goal['name'], 'estimated_error_m': distance, 'purpose': 'transit' if transit else 'arrival',
                        'estimated_tolerance_m': tolerance, 'stopped_left_m': stopped_left,
                        'reference_pose': list(pose), 'target_pose': [goal['x'], goal['y'], heading],
                        'odom_pose': list(motion.odom()),
                        'estimated_heading_error_deg': math.degrees(angle), 'reference_quality': quality,
                        'corrections': attempt, 'duration_s': time.monotonic() - began,
                        'physical_accuracy_verified': False}
            motion.report(evidence)
            if distance <= tolerance and abs(angle) <= math.radians(self.settings['heading_tolerance_deg']):
                return evidence
            if distance > self.settings['max_correction_distance']:
                raise RuntimeError('记录激光估计距离超过到点调整范围，车辆保持停车：%.1f毫米' % (distance * 1000))
            if attempt == self.settings['max_corrections']:
                if transit:
                    raise RuntimeError('路线中间点未达到通行要求，车辆保持停车：%.1f毫米' % (distance * 1000))
                raise RuntimeError('未达到设置的到点容差，车辆保持停车；估计距离%.1f毫米' % (distance * 1000))
            anchor = motion.odom()
            c, s = math.cos(pose[2]), math.sin(pose[2])
            body_error = np.array([c * error[0] + s * error[1], -s * error[0] + c * error[1]])
            # 每次只修正一个方向，使用反馈距离结束动作。
            if abs(angle) > math.radians(self.settings['heading_tolerance_deg']):
                desired, axis, amount = (0, 0, math.copysign(self.settings['turn_speed'], angle)), 2, angle
            else:
                axis = int(np.argmax(np.abs(body_error)))
                amount = float(body_error[axis])
                desired = [0.0, 0.0, 0.0]
                desired[axis] = math.copysign(self.settings['correction_speed'], amount)
            duration = self.settings['motion_timeout'] if axis != 2 else abs(amount) / self.settings['turn_speed'] + 2
            deadline = time.monotonic() + duration
            previous = 0.0
            started = False
            stopped_anchor = anchor
            try:
                while time.monotonic() < deadline:
                    motion.check()
                    actual = motion.odom()
                    velocity = motion.velocity()
                    threshold = 0.01 if axis != 2 else 0.05
                    if not started:
                        if abs(velocity[axis]) > threshold:
                            if velocity[axis] * amount <= 0:
                                raise RuntimeError('到点调整的速度反馈方向与指令不一致')
                            anchor = stopped_anchor
                            started = True
                        else:
                            stopped_anchor = actual
                            motion.send(*desired, remaining_distance=abs(amount) if axis != 2 else None)
                            time.sleep(0.03)
                            continue
                    dx, dy = actual[0] - anchor[0], actual[1] - anchor[1]
                    ca, sa = math.cos(anchor[2]), math.sin(anchor[2])
                    progress = (ca * dx + sa * dy, -sa * dx + ca * dy, wrap(actual[2] - anchor[2]))[axis]
                    if math.hypot(dx, dy) > self.settings['max_correction_distance']:
                        raise RuntimeError('到点调整期间移动超过允许范围')
                    # 里程计比车身实际动作滞后，停车指令发出后还会继续走一段，按当前速度提前收手。
                    lead = abs(velocity[axis]) * self.settings['latch_latency']
                    reserve = 0.0005 if axis != 2 else math.radians(0.5)
                    if amount * progress >= 0 and abs(progress) + lead >= abs(amount) - reserve:
                        break
                    reverse_limit = self.settings['stable_position'] if axis != 2 else math.radians(self.settings['stable_heading_deg'])
                    if (progress * amount < 0 and abs(progress) > reverse_limit) or abs(progress - previous) > (0.03 if axis != 2 else 0.3):
                        raise RuntimeError('到点调整的里程计方向或更新不连续')
                    previous = progress
                    motion.send(*desired, remaining_distance=max(0, abs(amount) - abs(progress)) if axis != 2 else None)
                    time.sleep(0.03)
                else:
                    raise RuntimeError('到点调整超过允许时间')
            finally:
                motion.stop()


class MonitorMotion:
    def __init__(self, monitor, navigator, driver=None, planner=None):
        self.monitor, self.navigator, self.driver, self.planner = monitor, navigator, driver, planner
        directory = Path.home() / 'final_ws/data/record/navigation_results'
        directory.mkdir(parents=True, exist_ok=True)
        self.audit_path = directory / ('position_checks_%d.jsonl' % time.time_ns())
        self.reconciled_at = {}
        self.initial_publisher = None
        self.map_matcher = None

    def verify_geometry(self, expected):
        import rospy
        import tf

        offset, quaternion = self.monitor.tf_listener.lookupTransform('base_footprint', 'laser_link', rospy.Time(0))
        yaw = tf.transformations.euler_from_quaternion(quaternion)[2]
        if math.dist(offset[:2], expected[:2]) > 0.001 or abs(wrap(yaw - expected[2])) > math.radians(0.1):
            raise RuntimeError('当前雷达TF与点位转换参数不一致，车辆保持停车')

    def reconcile(self, goal, pose, quality, points):
        import rospy
        from geometry_msgs.msg import PoseWithCovarianceStamped

        mapped = quality['mapped_pose']
        if math.dist(pose[:2], mapped[:2]) <= 0.03 and abs(wrap(pose[2] - mapped[2])) <= math.radians(2):
            return
        if time.monotonic() - self.reconciled_at.get(goal['name'], -math.inf) < 10:
            return
        if quality['inlier_fraction'] < 0.90:
            quality['amcl_update_eligible'] = False
            return
        if self.map_matcher is None:
            with self.monitor.lock:
                grid = self.monitor.map_msg
            if grid is None:
                raise RuntimeError('定位复核缺少地图')
            q = grid.info.origin.orientation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            self.map_matcher = MapMatcher(np.asarray(grid.data).reshape(grid.info.height, grid.info.width),
                                          grid.info.resolution, (grid.info.origin.position.x, grid.info.origin.position.y, yaw))
        independent = self.map_matcher.project({'name': goal['name'], 'x': pose[0], 'y': pose[1],
                                               'yaw_deg': math.degrees(pose[2]), 'scan_base': points})
        if math.dist(pose[:2], (independent['x'], independent['y'])) > 0.03 or abs(wrap(pose[2] - independent['yaw_rad'])) > math.radians(2):
            raise RuntimeError('记录激光与地图独立匹配结果不一致，车辆保持停车')
        if self.initial_publisher is None:
            self.initial_publisher = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=1)
            deadline = time.monotonic() + 2
            while self.initial_publisher.get_num_connections() == 0 and time.monotonic() < deadline:
                self.check()
                time.sleep(0.03)
            if self.initial_publisher.get_num_connections() == 0:
                raise RuntimeError('定位节点没有接收位置复核话题')
        message = PoseWithCovarianceStamped()
        message.header.frame_id = 'map'
        message.header.stamp = rospy.Time.now()
        message.pose.pose.position.x, message.pose.pose.position.y = pose[:2]
        message.pose.pose.orientation.z, message.pose.pose.orientation.w = math.sin(pose[2] / 2), math.cos(pose[2] / 2)
        message.pose.covariance[0] = message.pose.covariance[7] = 0.02 ** 2
        message.pose.covariance[35] = math.radians(2) ** 2
        self.initial_publisher.publish(message)
        self.reconciled_at[goal['name']] = time.monotonic()
        with self.navigator.lock:
            self.navigator.data['localization_reconciled'] = {'reference_pose': list(pose),
                'mapped_before': list(mapped), 'map_check': independent['projection_quality'], 'physical_accuracy_verified': False}
        rospy.loginfo('静止定位复核通过，使用记录激光与地图共同确认的位置更新AMCL')
        time.sleep(0.3)

    def check(self):
        import rospy

        active = self.navigator.state()['active']
        if rospy.is_shutdown() or (active and (self.navigator.cancel.is_set() or time.monotonic() - self.navigator.heartbeat_at > 2)):
            raise RuntimeError('导航已停止或控制连接超时')
        if not active:
            self.navigator.heartbeat_at = time.monotonic()
        self.navigator.check_sensors()

    def stationary_update(self):
        self.monitor.update_idle_localization(allow_navigation=True)

    def sample(self):
        self.check()
        with self.monitor.lock:
            scan, velocity, stamp = self.monitor.scan, self.monitor.vel, self.monitor.vel_stamp
        if scan is None or velocity is None or time.time() - stamp > 0.5:
            raise RuntimeError('停车检查缺少最新激光或速度反馈')
        points, pose = scan_sample(self.monitor.tf_listener, scan)
        return scan.header.stamp.to_nsec(), points, pose, velocity

    def odom(self):
        import rospy
        import tf

        listener = self.monitor.tf_listener
        stamp = listener.getLatestCommonTime('odom', 'base_footprint')
        if (rospy.Time.now() - stamp).to_sec() > 0.5:
            raise RuntimeError('到点调整期间里程计未及时更新')
        position, rotation = listener.lookupTransform('odom', 'base_footprint', rospy.Time(0))
        return position[0], position[1], tf.transformations.euler_from_quaternion(rotation)[2]

    def velocity(self):
        with self.monitor.lock:
            velocity, stamp = self.monitor.vel, self.monitor.vel_stamp
        if velocity is None or time.time() - stamp > 0.5:
            raise RuntimeError('到点调整的速度反馈未及时更新')
        return velocity

    def send(self, vx, vy, wz, remaining_distance=None):
        self.check()
        if self.planner is not None and (vx or vy):
            pose = self.monitor.pose()
            if pose is None:
                raise RuntimeError('到点调整缺少当前位置')
            c, s = math.cos(pose[2]), math.sin(pose[2])
            speed = math.hypot(vx, vy)
            distance = speed * 0.8 if remaining_distance is None else min(speed * 0.8, remaining_distance + 0.02)
            target = (pose[0] + (c * vx - s * vy) / speed * distance,
                      pose[1] + (s * vx + c * vy) / speed * distance)
            if not self.planner.visible(pose[:2], target):
                raise RuntimeError('到点调整的地图通行检查未通过')
        if self.driver is None:
            self.monitor.send_cmd(vx, vy, wz)
        else:
            self.driver.send(vx, vy, wz)

    def stop(self):
        if self.driver is None:
            self.monitor.stop()
        else:
            self.driver.stop(3)

    def report(self, evidence):
        import rospy

        evidence = {**evidence, 'stamp': time.time()}
        with self.audit_path.open('a', encoding='utf-8') as target:
            target.write(json.dumps(evidence, ensure_ascii=False, allow_nan=False) + '\n')
        with self.navigator.lock:
            self.navigator.data['precision'] = evidence
            self.navigator.data['distance'] = evidence['estimated_error_m']
            self.navigator.data['distance_source'] = 'recorded_laser'
            self.navigator.data['position_audit'] = str(self.audit_path)
        rospy.loginfo('记录点位核验：%s，估计距离%.1f毫米，朝向差%.2f度', evidence['goal'],
                      evidence['estimated_error_m'] * 1000, evidence['estimated_heading_error_deg'])
