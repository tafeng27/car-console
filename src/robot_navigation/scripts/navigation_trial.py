import argparse
import math
from pathlib import Path
import time
import subprocess

from navigation_plan import describe, inspect_plan, load_plan
from navigation_geometry import point_heading, prepare_points
from precision_navigation import MonitorMotion, PrecisionStop


def execute(config, points, selected, control=None):
    import rospy
    import tf
    import route_drive
    from sensor_msgs.msg import LaserScan

    if not rospy.core.is_initialized():
        rospy.init_node("navigation_trial")
    limits = config["limits"]
    settings = {"speed": limits["forward"], "reverse_speed": limits["strafe"],
                "turn_speed": limits["angular"], "corner_turn_speed": limits["angular"],
                "accel": limits["acceleration"], "decel": limits["deceleration"],
                "min_speed": 0.12, "lat_max": 0.10, "pos_tol": limits["position_tolerance"]}
    for name, value in settings.items():
        rospy.set_param("~" + name, value)

    class Driver(route_drive.Driver):
        def __init__(self):
            super().__init__()
            self.scan = None
            self.last_command = (0.0, 0.0, 0.0)
            self.last_send = time.monotonic()
            self.sensor_stop_epoch = 0
            self.target = None
            self.segment_start = None
            self.stopped_left = None
            self.scan_subscriber = rospy.Subscriber("/scan", LaserScan, self.on_scan, queue_size=1)

        def check_control(self):
            if control is not None:
                if control.cancel.is_set():
                    raise RuntimeError("路线导航已停止")
                if time.monotonic() - control.heartbeat_at > 2:
                    raise RuntimeError("页面连接超过2秒未更新，路线导航已停止")
                control.check_sensors()
                if self.sensor_stop_epoch != control.sensor_stop_epoch:
                    self.last_command = (0.0, 0.0, 0.0)
                    self.last_send = time.monotonic()
                    self.sensor_stop_epoch = control.sensor_stop_epoch

        def on_scan(self, message):
            self.scan = message

        def pose(self):
            self.check_control()
            stamp = self.tf.getLatestCommonTime("map", "base_footprint")
            if (rospy.Time.now() - stamp).to_sec() > 0.5:
                raise RuntimeError("定位数据超过0.5秒未更新")
            pose = super().pose()
            if control is not None and getattr(control, 'route_planner', None) is not None:
                if not control.route_planner.valid(control.route_planner.cell(pose[:2])):
                    raise RuntimeError('当前位置不满足规定路线的车身通行范围')
            if self.target is not None:
                dx, dy = self.target[0] - self.segment_start[0], self.target[1] - self.segment_start[1]
                length = math.hypot(dx, dy)
                if length >= 0.02:
                    cross = ((pose[0] - self.segment_start[0]) * dy - (pose[1] - self.segment_start[1]) * dx) / length
                    if abs(cross) > 0.10:
                        raise RuntimeError("车身已离开计划线段超过10厘米")
            return pose

        def stop(self, n=5):
            self.last_command = (0.0, 0.0, 0.0)
            self.last_send = time.monotonic()
            super().stop(n)

        def send(self, vx, vy, wz):
            self.check_control()
            if self.scan is None or (rospy.Time.now() - self.scan.header.stamp).to_sec() > 0.5:
                raise RuntimeError("雷达数据超过0.5秒未更新")
            now = time.monotonic()
            dt = min(now - self.last_send, 0.1)
            angular_step = limits["angular_acceleration"] * dt
            wz = max(self.last_command[2] - angular_step, min(self.last_command[2] + angular_step, wz))
            wz = max(-limits["angular"], min(limits["angular"], wz))
            speed = math.hypot(vx, vy)
            if self.target is not None and speed > 0:
                x, y, _ = self.pose()
                ux, uy = self.target[0] - self.segment_start[0], self.target[1] - self.segment_start[1]
                length = math.hypot(ux, uy)
                remaining = ((self.target[0] - x) * ux + (self.target[1] - y) * uy) / length
                delay_term = limits["deceleration"] * limits["command_delay"]
                cap = max(0.12, math.sqrt(delay_term ** 2 + 2 * limits["deceleration"] * max(0, remaining - self.pos_tol)) - delay_term)
                scale = min(1.0, cap / speed, limits["forward"] / speed)
                vx, vy = vx * scale, vy * scale
            acceleration = limits["lateral_acceleration"] if abs(vy) > abs(vx) else limits["acceleration"]
            dx, dy = vx - self.last_command[0], vy - self.last_command[1]
            change = math.hypot(dx, dy)
            scale = min(1.0, acceleration * dt / change) if change else 1.0
            vx, vy = self.last_command[0] + dx * scale, self.last_command[1] + dy * scale
            self.last_command, self.last_send = (vx, vy, wz), now
            super().send(vx, vy, wz)

        def odom_pose(self):
            stamp = self.tf.getLatestCommonTime("odom", "base_footprint")
            if (rospy.Time.now() - stamp).to_sec() > 0.5:
                raise RuntimeError("里程计数据超过0.5秒未更新")
            location, rotation = self.tf.lookupTransform("odom", "base_footprint", rospy.Time(0))
            return location[0], location[1], tf.transformations.euler_from_quaternion(rotation)[2]

        def rotate(self, angle, timeout=20, w_max=None):
            anchor = self.odom_pose()
            map_anchor = self.pose()
            direction = 1 if angle >= 0 else -1
            remaining, previous = abs(angle), anchor[2]
            deadline = time.monotonic() + timeout
            active_translation = False
            while True:
                self.check_control()
                if rospy.is_shutdown() or time.monotonic() > deadline:
                    raise RuntimeError("连续转向控制超时")
                x, y, yaw = self.odom_pose()
                remaining -= direction * route_drive.wrap(yaw - previous)
                previous = yaw
                dx, dy = anchor[0] - x, anchor[1] - y
                distance = math.hypot(dx, dy)
                if distance > 0.10:
                    raise RuntimeError("转向期间车身位移超过10厘米")
                if remaining <= self.turn_tol and distance <= 0.025:
                    break
                # 连续修正实际位移，使用滞后区间保持指令连续。
                active_translation = distance > (0.015 if active_translation else 0.025)
                vx = vy = 0.0
                if active_translation:
                    c, s = math.cos(yaw), math.sin(yaw)
                    scale = min(0.18, max(0.14, distance * 2)) / distance
                    vx, vy = (c * dx + s * dy) * scale, (-s * dx + c * dy) * scale
                angular = 0.0 if remaining <= self.turn_tol else direction * max(
                    self.turn_w_min, min(self.w_max if w_max is None else w_max, self.turn_gain * remaining))
                self.send(vx, vy, angular)
                self.rate.sleep()
            self.stop(3)
            actual, mapped = self.odom_pose(), self.pose()
            angle_error = route_drive.wrap(actual[2] - anchor[2] - angle)
            rospy.loginfo("转向完成：odom位移%.3f米，map位置变化%.3f米，剩余角度%.1f度",
                          math.dist(actual[:2], anchor[:2]), math.dist(mapped[:2], map_anchor[:2]), math.degrees(remaining))
            if math.dist(actual[:2], anchor[:2]) > 0.04:
                raise RuntimeError("转向停车后的车身位移超过4厘米")
            if abs(angle_error) > math.radians(6):
                raise RuntimeError("转向停车后的朝向误差超过6度")

        def turn_to(self, yaw, timeout=20, w_max=None):
            angle = route_drive.wrap(yaw - self.pose()[2])
            if abs(angle) > self.turn_tol:
                self.rotate(angle, timeout, w_max)

        def turn_relative(self, degrees):
            self.rotate(math.radians(degrees))

        def adjust_left(self, distance):
            anchor = self.odom_pose()
            target = (anchor[0] - math.sin(anchor[2]) * distance, anchor[1] + math.cos(anchor[2]) * distance)
            deadline = time.monotonic() + 8
            try:
                while True:
                    x, y, yaw = self.odom_pose()
                    dx, dy = target[0] - x, target[1] - y
                    remaining = math.hypot(dx, dy)
                    if remaining <= 0.006:
                        break
                    if time.monotonic() > deadline or math.dist((x, y), anchor[:2]) > 0.06:
                        raise RuntimeError("点位小范围修正未在允许范围完成")
                    c, s = math.cos(yaw), math.sin(yaw)
                    self.send(0.14 * (c * dx + s * dy) / remaining, 0.14 * (-s * dx + c * dy) / remaining, 0)
                    self.rate.sleep()
            finally:
                self.stop(3)
            rospy.loginfo("已确认位置修正：向车头左侧移动%.3f米", math.dist(self.odom_pose()[:2], anchor[:2]))

        def move(self, point, mode, final=False, speed=None):
            started = time.monotonic()
            x, y, yaw = self.pose()
            self.segment_start = (x, y)
            target = (point['x'], point['y'])
            if control is not None and not control.route_planner.visible(self.segment_start, target):
                finish_point(points[current_point], yaw, corridor_entry=True)
                x, y, yaw = self.pose()
                self.segment_start = (x, y)
                if not control.route_planner.visible(self.segment_start, target):
                    raise RuntimeError('当前位置到下一规定路线点的通行检查未通过')
            self.target = target
            self.pos_tol = min(limits["final_tolerance"] if final else limits["position_tolerance"],
                               (min(self.v_min, limits["final_reverse"]) if final else self.v_min) * limits["stop_latency"])
            self.v_end = limits["final_reverse"] if final else limits[mode if mode == "reverse" else "strafe"]
            try:
                self.go(self.segment_start, self.target, reverse=mode == "reverse",
                        body=yaw if mode in ("left", "right") else None,
                        slow=mode in ("left", "right"), v_cap=speed)
                x, y, _ = self.pose()
                error = math.hypot(x - point["x"], y - point["y"])
                ux, uy = target[0] - self.segment_start[0], target[1] - self.segment_start[1]
                length = math.hypot(ux, uy)
                self.stopped_left = ((target[0] - x) * ux + (target[1] - y) * uy) / length if length >= 0.02 else 0.0
                rospy.loginfo("行驶完成：%s，耗时%.2f秒，目标距离%.3f米，停车时沿段剩余%.1f毫米", mode,
                              time.monotonic() - started, error, self.stopped_left * 1000)
                allowed_error = 0.12
                if error > allowed_error:
                    raise RuntimeError("行驶未到达目标点位")
            finally:
                self.target = None
                self.stop(3)
            finish_point(point, self.pose()[2], stopped_left=self.stopped_left)

    driver = Driver()
    if control is None:
        raise RuntimeError('精确导航需要通过控制台或start.sh路线入口启动')
    precision = PrecisionStop(MonitorMotion(control.monitor, control, driver=driver,
                                           planner=control.route_planner), config['precision'])

    def finish_point(point, heading, corridor_entry=False, stopped_left=None):
        precision.finish(point, heading, transit='single_point_target' in config,
                         transit_tolerance=0.005 if corridor_entry else 0.02, stopped_left=stopped_left)
    rospy.sleep(1.0)
    others = route_drive.other_cmd_vel_publishers()
    if others:
        raise RuntimeError("请释放其他驾驶发布者：" + ", ".join(others))
    if not rospy.has_param("/amcl/initial_pose_x"):
        raise RuntimeError("请先启动 AMCL 定位")
    start = points[config["start"]]
    x, y, yaw = driver.pose()
    start_tolerance = float(config.get("start_tolerance", 0.15))
    if math.hypot(x - start["x"], y - start["y"]) > start_tolerance or abs(route_drive.wrap(yaw - math.radians(config.get("continuation_heading", start["yaw_deg"])))) > math.radians(15):
        raise RuntimeError("请将小车定位到起停点，并保持记录的起始朝向")

    from race_vision import RaceVision

    vision = None
    current_point = config["start"]

    def perform_task(name):
        driver.stop(3)
        finish_point(points[current_point], driver.pose()[2])
        started = time.monotonic()
        if vision is not None and name in ("crowd_A", "crowd_B"):
            rospy.sleep(0.25)
            vision.record_view(name[-1], current_point)
        elif vision is not None and name == "traffic_light_1":
            vision.wait_green(driver.stop)
        elif vision is not None and name == "trash":
            vision.record_trash(current_point)
        elif vision is not None and name in ("building_A", "building_B", "building_C"):
            rospy.sleep(0.25)
            vision.record_building(name[-1], current_point)
        elif vision is not None and name == "ebike":
            rospy.sleep(0.25)
            vision.record_parking_view(current_point, driver.pose)
        else:
            rospy.loginfo("识别预留：%s，停车2秒", name)
        while time.monotonic() - started < config["task_dwell"]:
            driver.check_control()
            rospy.sleep(0.05)

    def run(steps, active_route=None):
        nonlocal current_point
        for index, step in enumerate(steps, 1):
            if rospy.is_shutdown():
                raise RuntimeError("导航已终止")
            rospy.loginfo("动作 %d：%s", index, step)
            if control is not None:
                with control.lock:
                    control.data["message"] = "路线动作%d：%s" % (index, step)
                    if "move" in step:
                        control.data["goal"] = step["move"]
            if "move" in step:
                driver.move(points[step["move"]], step["mode"], step.get("final", False), step.get("speed"))
                current_point = step["move"]
            elif "turn" in step:
                degrees = step["route2_turn"] if active_route == 2 and "route2_turn" in step else step["turn"]
                heading = route_drive.wrap(driver.pose()[2] + math.radians(degrees))
                driver.turn_relative(degrees)
                finish_point(points[current_point], heading)
            elif "face" in step:
                driver.turn_to(math.radians(step["face"]))
                finish_point(points[current_point], math.radians(step['face']))
            elif "face_point" in step:
                heading = point_heading(config, points, step["face_point"]) + step.get("offset", 0)
                driver.turn_to(math.radians(heading))
                finish_point(points[current_point], math.radians(heading))
            else:
                if "look" in step:
                    driver.turn_to(math.radians(float(step["look"])))
                    finish_point(points[current_point], math.radians(float(step['look'])))
                perform_task(step["task"])
            if control is not None and '_route_cursor_after' in step:
                with control.lock:
                    control.data['route_cursor'] = step['_route_cursor_after']
                    control.data['route_anchor'] = current_point

    def choose_route():
        gate = config["branch"]
        origin, toward = points[gate["origin"]], points[gate["toward"]]
        heading = math.atan2(toward["y"] - origin["y"], toward["x"] - origin["x"])
        decisions = []
        for _ in range(3):
            driver.check_control()
            scan = rospy.wait_for_message("/scan", LaserScan, timeout=2)
            driver.tf.waitForTransform("map", scan.header.frame_id, scan.header.stamp, rospy.Duration(2))
            location, rotation = driver.tf.lookupTransform("map", scan.header.frame_id, scan.header.stamp)
            angle = tf.transformations.euler_from_quaternion(rotation)[2]
            count = 0
            for index, distance in enumerate(scan.ranges):
                if not max(scan.range_min, gate["min_distance"]) <= distance <= min(scan.range_max, 2.0):
                    continue
                beam = scan.angle_min + index * scan.angle_increment + angle
                dx = location[0] + distance * math.cos(beam) - origin["x"]
                dy = location[1] + distance * math.sin(beam) - origin["y"]
                along = dx * math.cos(heading) + dy * math.sin(heading)
                across = -dx * math.sin(heading) + dy * math.cos(heading)
                count += int(gate["min_distance"] <= along <= gate["length"] and abs(across) < gate["width"] / 2)
            decisions.append(count >= gate["min_points"])
            rospy.loginfo("岔路激光点数：%d", count)
        return gate["blocked_route"] if sum(decisions) >= 2 else gate["clear_route"]

    completed = False
    try:
        if config["recognition"]["enabled"]:
            vision = RaceVision(config["recognition"], control, driver.check_control)
        if config.get("left_adjustment", 0):
            driver.adjust_left(config["left_adjustment"])
        run(config["common"])
        route = choose_route() if selected == "auto" else int(selected)
        rospy.loginfo("执行路线%d", route)
        run(config["route" + str(route)])
        run(config["tail"], route)
        if 'single_point_target' in config:
            precision.finish(points[config['single_point_target']], config['single_point_final_heading'])
        completed = True
    finally:
        driver.stop(10)
        if vision is not None:
            vision.close(completed)
        driver.scan_subscriber.unregister()
        if control is not None:
            driver.pub.unregister()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "config/navigation_trial.yaml"))
    parser.add_argument("--route", choices=("auto", "1", "2"), default="auto")
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    config, points = load_plan(arguments.config)
    problems = inspect_plan(config, points)
    if arguments.check:
        describe(config, points)
    if problems:
        raise SystemExit("\n".join(problems))
    if not arguments.check:
        workspace = Path(__file__).resolve().parents[3]
        canonical = workspace / 'src/robot_navigation/config/navigation_trial.yaml'
        if Path(arguments.config).resolve() != canonical:
            raise SystemExit('终端路线入口需要使用工程当前导航配置')
        subprocess.run(['/usr/bin/python3', str(workspace / 'tools/route_entry.py'), '--route', arguments.route], check=True)
