import copy
import heapq
import math
import statistics
import threading
import time

import cv2
import numpy as np
from localization_quality import require_quality, scan_quality


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class GridPlanner:
    def __init__(self, cells, resolution, origin, radius=0.235):
        self.resolution = resolution
        self.origin = origin
        self.radius = radius
        # 未知区域参与障碍扩张，车身使用完整旋转范围。车身外接半径 0.225 米，另留 1 厘米余量。
        free = ((np.asarray(cells) >= 0) & (np.asarray(cells) <= 20)).astype(np.uint8)
        padded = np.pad(free, 1)
        distance = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
        self.clearance = distance * resolution - resolution * math.sqrt(2) / 2
        self.free = self.clearance >= radius

    def cell(self, point):
        dx, dy = point[0] - self.origin[0], point[1] - self.origin[1]
        c, s = math.cos(self.origin[2]), math.sin(self.origin[2])
        return (math.floor((-dx * s + dy * c) / self.resolution),
                math.floor((dx * c + dy * s) / self.resolution))

    def world(self, cell):
        x, y = (cell[1] + 0.5) * self.resolution, (cell[0] + 0.5) * self.resolution
        c, s = math.cos(self.origin[2]), math.sin(self.origin[2])
        return self.origin[0] + x * c - y * s, self.origin[1] + x * s + y * c

    def valid(self, cell):
        return 0 <= cell[0] < self.free.shape[0] and 0 <= cell[1] < self.free.shape[1] and bool(self.free[cell])

    def visible(self, first, second, margin=0.0):
        count = max(1, math.ceil(math.dist(first, second) / (self.resolution / 4)))
        length = math.dist(first, second)
        for index in range(count + 1):
            cell = self.cell((first[0] + (second[0] - first[0]) * index / count,
                              first[1] + (second[1] - first[1]) * index / count))
            extra = margin * min(1.0, length * index / count / 0.10, length * (count - index) / count / 0.10)
            if not self.valid(cell) or self.clearance[cell] < self.radius + extra:
                return False
        return True

    def plan(self, start, goal):
        first, last = self.cell(start), self.cell(goal)
        if not self.valid(first):
            raise ValueError("当前位置距离地图障碍过近，无法规划")
        if not self.valid(last):
            raise ValueError("目标点位距离地图障碍过近，请调整点位")
        queue, costs, previous = [(0.0, first)], {first: 0.0}, {}
        while queue:
            _, current = heapq.heappop(queue)
            if current == last:
                break
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                following = (current[0] + dr, current[1] + dc)
                if not self.valid(following):
                    continue
                if dr and dc and (not self.valid((current[0] + dr, current[1])) or not self.valid((current[0], current[1] + dc))):
                    continue
                cost = costs[current] + math.hypot(dr, dc) * (1 + 0.04 / max(0.01, self.clearance[following] - self.radius))
                if cost < costs.get(following, math.inf):
                    costs[following], previous[following] = cost, current
                    heapq.heappush(queue, (cost + math.dist(following, last), following))
        if last not in costs:
            raise ValueError("当前地图中没有满足车身通行距离的路线")
        cells = [last]
        while cells[-1] != first:
            cells.append(previous[cells[-1]])
        raw = [tuple(start)] + [self.world(cell) for cell in reversed(cells)] + [tuple(goal)]
        path, index = [raw[0]], 0
        while index < len(raw) - 1:
            following = len(raw) - 1
            while following > index + 1 and not self.visible(raw[index], raw[following], margin=0.01):
                following -= 1
            if following == index or not self.visible(raw[index], raw[following]):
                raise ValueError("路线连续性检查未通过")
            path.append(raw[following])
            index = following
        return path


class PointNavigator:
    def __init__(self, monitor):
        self.monitor = monitor
        self.lock = threading.RLock()
        self.lifecycle_lock = threading.RLock()
        self.thread = None
        self.cancel = threading.Event()
        self.heartbeat_at = 0.0
        self.last_alignment = None
        self.last_quality = None
        self.sensor_stop_epoch = 0
        self.trash_preview = None
        self.data = {"active": False, "status": "idle", "message": "请选择已记录点位", "goal": None, "path": []}

    def state(self):
        with self.lock:
            return {**self.data, "localization": self.last_quality}

    def heartbeat(self):
        with self.lock:
            self.heartbeat_at = time.monotonic()

    def stop(self):
        with self.lifecycle_lock:
            self._stop()

    def _stop(self):
        with self.lock:
            thread = self.thread
            self.cancel.set()
        if thread and thread.is_alive():
            thread.join(3)
            if thread.is_alive():
                raise RuntimeError("单点导航尚未停止")

    def start(self, goal, heading=None, source=None, cursor=None):
        with self.lifecycle_lock:
            return self._start(goal, heading, source, cursor)

    def make_route_planner(self, points):
        from precision_navigation import navigation_cells

        with self.monitor.lock:
            grid = self.monitor.map_msg
        if grid is None:
            raise ValueError('路线检查缺少地图')
        q = grid.info.origin.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        origin = (grid.info.origin.position.x, grid.info.origin.position.y, yaw)
        cells, removed = navigation_cells(np.asarray(grid.data).reshape(grid.info.height, grid.info.width),
                                           grid.info.resolution, origin, points)
        return GridPlanner(cells, grid.info.resolution, origin), removed

    def start_rotation(self, degrees):
        import rospy

        if not math.isfinite(degrees) or not 5 <= abs(degrees) <= 180:
            raise ValueError('转向采样角度需要为5至180度')
        with self.lifecycle_lock:
            self.stop()
            if self.monitor.mode() != 'localize':
                raise ValueError('转向采样需要定位状态')
            self.cancel = threading.Event()
            self.heartbeat_at = time.monotonic()
            self.check_sensors()
            with self.monitor.lock:
                grid = self.monitor.map_msg
            if grid is None:
                raise ValueError('转向采样缺少地图')
            q = grid.info.origin.orientation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            planner = GridPlanner(np.asarray(grid.data).reshape(grid.info.height, grid.info.width), grid.info.resolution,
                                  (grid.info.origin.position.x, grid.info.origin.position.y, yaw))
            pose = self.monitor.pose()
            if pose is None or not planner.valid(planner.cell(pose[:2])):
                raise ValueError('当前地图通行范围不允许原地转向')
            self.data = {'active': True, 'status': 'running', 'message': '正在进行原地转向采样', 'goal': None, 'path': []}
            self.thread = threading.Thread(target=self.run_rotation, args=(degrees,), daemon=True)
            self.thread.start()
        return self.state()

    def run_rotation(self, degrees):
        from precision_navigation import MonitorMotion

        motion = MonitorMotion(self.monitor, self)
        result, message = 'arrived', '原地转向采样完成'
        try:
            anchor = motion.odom()
            previous, remaining = anchor[2], math.radians(abs(degrees))
            direction = 1 if degrees > 0 else -1
            deadline = time.monotonic() + remaining / 0.4 + 3
            while remaining > math.radians(0.5):
                motion.check()
                actual = motion.odom()
                if math.dist(actual[:2], anchor[:2]) > 0.02 or time.monotonic() > deadline:
                    raise RuntimeError('转向采样的位移或用时超过允许范围')
                remaining -= direction * wrap(actual[2] - previous)
                previous = actual[2]
                if remaining <= math.radians(0.5):
                    break
                motion.send(0, 0, direction * 0.4)
                time.sleep(0.03)
        except Exception as error:
            result, message = 'error', str(error)
        finally:
            motion.stop()
            with self.lock:
                self.data.update(active=False, status=result, message=message)

    def prepare_route(self, selected, resume=None):
        from pathlib import Path
        from navigation_geometry import prepare_points
        from navigation_plan import inspect_plan, load_plan

        if selected not in ("auto", "1", "2"):
            raise ValueError("请选择 auto、1 或 2")
        if self.monitor.mode() != "localize":
            raise ValueError("路线导航需要 AMCL 定位")
        with self.lock:
            self.heartbeat_at = time.monotonic()
        self.check_sensors()
        path = Path(__file__).resolve().parent.parent / "config/navigation_trial.yaml"
        config, points = load_plan(path)
        problems = inspect_plan(config, points)
        if problems:
            raise ValueError("；".join(problems))
        points = prepare_points(config, points, path)
        self.route_planner, _ = self.make_route_planner(points)
        pose = self.monitor.pose()
        start = points[resume["start"] if resume else config["start"]]
        start_tolerance = float(resume.get("start_tolerance", 0.15)) if resume else 0.15
        if resume is not None and not 0.15 <= start_tolerance <= 0.30:
            raise ValueError("继续行驶的起点距离容差需要为15至30厘米")
        if pose is None or math.dist(pose[:2], (start["x"], start["y"])) > start_tolerance or (resume is None and abs(wrap(pose[2] - math.radians(start["yaw_deg"]))) > math.radians(15)):
            raise ValueError("请将小车放回起停点并保持起始朝向")
        import rospy
        from sensor_msgs.msg import LaserScan

        samples = [self.alignment_details(rospy.wait_for_message("/scan", LaserScan, timeout=2)) for _ in range(3)]
        if any(sample["wall_percent"] is None for sample in samples):
            raise ValueError("没有取得足够的墙面激光数据")
        quality = {key: statistics.median(sample[key] for sample in samples) for key in samples[0] if key != "coverage_ok"}
        quality["coverage_ok"] = all(sample["coverage_ok"] for sample in samples)
        with self.lock:
            self.last_quality = quality
            self.last_alignment = quality["wall_percent"]
        if resume is None:
            require_quality(quality)
            from precision_navigation import MonitorMotion, PrecisionStop
            stopped_pose, reference_quality = PrecisionStop(MonitorMotion(self.monitor, self), config['precision']).observe(start)
            reference_distance = math.dist(stopped_pose[:2], (start['x'], start['y']))
            if reference_distance > 0.03 or abs(wrap(stopped_pose[2] - math.radians(start['yaw_deg']))) > math.radians(5):
                raise ValueError('起停点记录激光检查未通过：距离%.1f毫米，车辆保持停车' % (reference_distance * 1000))
            with self.lock:
                self.last_quality = {**quality, 'reference_distance_m': reference_distance,
                                     'reference_quality': reference_quality, 'physical_accuracy_verified': False}
        elif not quality["coverage_ok"] or quality["wall_percent"] < 95:
            raise ValueError("继续行驶需要两个方向墙面贴合率达到95%")
        if resume is not None:
            phase, index = resume["phase"], resume["index"]
            if phase not in ("common", "route" + selected, "tail"):
                raise ValueError("继续位置与所选路线不一致")
            if not 0 <= index < len(config[phase]):
                raise ValueError("继续动作超出路线范围")
            action = config[phase][index]
            if action.get("move") != resume["goal"] and not ("turn" in action and "heading" in resume):
                raise ValueError("继续位置需要对应已确认的行驶或转向动作")
            config = copy.deepcopy(config)
            if phase == "common":
                config["common"] = config["common"][index:]
            else:
                config["common"] = []
                if phase == "tail":
                    config["route" + selected] = []
                    config["tail"] = config["tail"][index:]
                else:
                    config[phase] = config[phase][index:]
            if "turn" in action:
                heading = float(resume["heading"])
                change = wrap(math.radians(heading) - pose[2])
                if not -180 <= heading <= 180 or change * action["turn"] < 0 or abs(change) > math.radians(abs(action["turn"]) + 6):
                    raise ValueError("剩余转向与原路线方向不一致")
                config[phase][0] = {"face": heading}
            config["start"] = resume["start"]
            config["start_tolerance"] = start_tolerance
            config["continuation_heading"] = math.degrees(pose[2])
            adjustment = float(resume.get("left_adjustment", 0))
            if not 0 <= adjustment <= 0.05:
                raise ValueError("确认的位置修正最多5厘米")
            config["left_adjustment"] = adjustment
            config["recognition"]["resume_result"] = resume["result_file"]
        return config, points

    def map_alignment(self, scan=None):
        return self.alignment_details(scan)["all_percent"]

    def alignment_details(self, scan=None):
        with self.monitor.lock:
            grid = self.monitor.map_msg
            if scan is None:
                scan = self.monitor.scan
        if grid is None or scan is None:
            raise ValueError("尚未取得地图或雷达数据")
        cells = np.asarray(grid.data).reshape(grid.info.height, grid.info.width)
        if not np.any(cells >= 65):
            raise ValueError("地图没有占用区域")
        resolution = grid.info.resolution
        position, quaternion = self.monitor.tf_listener.lookupTransform("map", scan.header.frame_id, scan.header.stamp)
        x, y, z, w = quaternion
        heading = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        ranges = np.asarray(scan.ranges)
        valid = (ranges > max(scan.range_min, 0.15)) & (ranges < min(scan.range_max, 6))
        angles = scan.angle_min + np.arange(len(ranges))[valid] * scan.angle_increment + heading
        origin = grid.info.origin
        q = origin.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        dx, dy = position[0] + ranges[valid] * np.cos(angles) - origin.position.x, position[1] + ranges[valid] * np.sin(angles) - origin.position.y
        local_x, local_y = dx * math.cos(yaw) + dy * math.sin(yaw), -dx * math.sin(yaw) + dy * math.cos(yaw)
        if not len(local_x):
            raise ValueError("没有有效的激光数据")
        return scan_quality(cells, resolution, np.column_stack((local_x, local_y)))

    def start_route(self, selected, resume=None):
        with self.lifecycle_lock:
            self.stop()
            config, points = self.prepare_route(selected, resume)
            self.monitor.stop()
            with self.lock:
                self.cancel = threading.Event()
                self.heartbeat_at = time.monotonic()
                self.data = {"active": True, "status": "running", "message": "路线导航已启动",
                             "goal": config["start"], "path": [], "route": selected}
                self.trash_preview = None
                self.thread = threading.Thread(target=self.run_route, args=(config, points, selected), daemon=True, name="route_navigation")
                self.thread.start()
            return self.state()

    def run_route(self, config, points, selected):
        from navigation_trial import execute

        result, message = "arrived", "路线已完成，小车已返回起停点"
        try:
            execute(config, points, selected, control=self)
        except Exception as error:
            result, message = ("stopped", "路线导航已停止") if self.cancel.is_set() else ("error", str(error))
        finally:
            self.monitor.stop()
            with self.lock:
                self.data.update(active=False, status=result, message=message)

    def _start(self, goal, heading=None, source=None, cursor_override=None):
        import rosgraph
        import rospy
        from pathlib import Path
        from navigation_geometry import prepare_points
        from navigation_plan import load_plan
        from precision_navigation import MonitorMotion, PrecisionStop
        from route_navigation import navigation_slice, route_nodes

        previous = self.state()
        self.stop()
        publishers, _, _ = rosgraph.Master(rospy.get_name()).getSystemState()
        others = [node for topic, nodes in publishers if topic == "/cmd_vel" for node in nodes if node != rospy.get_name()]
        if others:
            raise ValueError("请停止其他驾驶程序：" + ", ".join(others))
        if self.monitor.mode() != "localize":
            raise ValueError("单点导航需要 AMCL 定位")
        config_path = Path(__file__).resolve().parent.parent / 'config/navigation_trial.yaml'
        config, recorded = load_plan(config_path)
        projected = prepare_points(config, recorded, config_path)
        goal = projected[goal['name']]
        if heading is not None:
            if not math.isfinite(heading) or not -180 <= heading <= 180:
                raise ValueError('目标朝向需要为负180至180度')
            goal['arrival_heading'] = heading
        route = config['single_point_route']
        _, nodes = route_nodes(config, projected, route)
        allowed = {node['name'] for node in nodes}
        if goal['name'] not in allowed:
            raise ValueError('该点位未加入路线%s，不能自动前往' % route)
        pose = self.monitor.pose()
        if pose is None:
            raise ValueError("尚未取得当前定位")
        session = goal['recording_session']
        cursor = None
        if source is None and previous.get('route_mode') == 'ordered' and previous.get('route_session') == session:
            source, cursor = previous.get('route_anchor'), previous.get('route_cursor')
        if source is None:
            candidates = sorted((math.dist(pose[:2], (projected[name]['x'], projected[name]['y'])), name) for name in allowed)
            if candidates[0][0] > 0.20 or (len(candidates) > 1 and candidates[1][0] - candidates[0][0] < 0.03):
                raise ValueError('当前所在路线点位无法唯一确认，请停在已记录路线点位后重试')
            source = candidates[0][1]
        if source not in allowed:
            raise ValueError('当前点位未加入规定路线')
        if cursor_override is not None:
            if not isinstance(cursor_override, int) or not any(node['name'] == source and node['begin'] <= cursor_override <= node['end'] for node in nodes):
                raise ValueError('恢复动作与当前路线点位不一致')
            cursor = cursor_override
        planner, removed_self_cells = self.make_route_planner(projected)
        self.monitor.stop()
        self.cancel = threading.Event()
        self.heartbeat_at = time.monotonic()
        self.check_sensors()
        motion = MonitorMotion(self.monitor, self, planner=planner)
        current, _ = PrecisionStop(motion, config['precision']).observe(projected[source])
        if math.dist(current[:2], (projected[source]['x'], projected[source]['y'])) > 0.10:
            raise ValueError('车辆没有停在确认的路线点位附近，禁止自动连接路线')
        selected, names = navigation_slice(config, projected, route, source, goal['name'], cursor, heading)
        selected['continuation_heading'] = math.degrees(self.monitor.pose()[2])
        path = [tuple(current[:2])] + [(projected[name]['x'], projected[name]['y']) for name in names]
        for first, last in zip(path, path[1:]):
            if not planner.visible(first, last):
                raise ValueError('规定路线段通行检查未通过，车辆保持停车')
        self.route_planner = planner
        with self.lock:
            self.cancel = threading.Event()
            self.heartbeat_at = time.monotonic()
            self.data = {"active": True, "status": "running", "message": "按规定路线前往 " + goal["name"],
                         "goal": goal["name"], "path": path, "segment": 0,
                         'route_mode': 'ordered', 'route': str(route), 'route_session': session,
                         'requested_goal': goal['name'], 'route_anchor': source, 'route_points': names,
                         'route_cursor': cursor,
                         'goal_pose': {key: goal[key] for key in ('x', 'y', 'yaw_deg')},
                         'recorded_pose': goal['recorded_pose'], 'excluded_self_cells': removed_self_cells}
            self.thread = threading.Thread(target=self.run_ordered, args=(selected, projected, route, goal['name']), daemon=True, name="point_navigation")
            self.thread.start()
        return self.state()

    def run_ordered(self, config, points, route, target):
        from navigation_trial import execute

        result, message = 'arrived', '已沿规定路线到达 ' + target + '，实际停车距离需测量'
        try:
            execute(config, points, str(route), control=self)
        except Exception as error:
            result, message = ('stopped', '单点导航已停止') if self.cancel.is_set() else ('error', str(error))
        finally:
            self.monitor.stop()
            with self.lock:
                self.data.update(active=False, status=result, message=message)

    def check_sensors(self):
        import rospy

        if not rospy.get_param("/imu_bias_filter/calibrated", False):
            raise RuntimeError("请等待IMU静止校准完成")
        stamp = self.monitor.tf_listener.getLatestCommonTime("map", "base_footprint")
        if (rospy.Time.now() - stamp).to_sec() > 0.5:
            raise RuntimeError("定位超过0.5秒未更新")
        with self.monitor.lock:
            scan, covariance = self.monitor.scan, self.monitor.amcl_pose
        if scan is None or (rospy.Time.now() - scan.header.stamp).to_sec() > 0.5:
            raise RuntimeError("雷达超过0.5秒未更新")
        deadline, update_at = time.monotonic() + 8, 0.0
        paused = False
        while covariance is None or max(covariance.pose.covariance[0], covariance.pose.covariance[7]) > 0.15 ** 2 or covariance.pose.covariance[35] > math.radians(12) ** 2:
            if not paused:
                self.sensor_stop_epoch += 1
                paused = True
            self.monitor.stop()
            active = self.state()['active']
            if (active and (self.cancel.is_set() or time.monotonic() - self.heartbeat_at > 2)) or time.monotonic() > deadline:
                raise RuntimeError("停车等待后定位分布仍未收敛，导航已停止")
            if time.monotonic() - update_at >= 1:
                self.monitor.idle_update()
                update_at = time.monotonic()
            time.sleep(0.05)
            with self.monitor.lock:
                covariance, scan = self.monitor.amcl_pose, self.monitor.scan
            if scan is None or (rospy.Time.now() - scan.header.stamp).to_sec() > 0.5:
                raise RuntimeError("等待定位期间雷达数据未更新")
        return scan
