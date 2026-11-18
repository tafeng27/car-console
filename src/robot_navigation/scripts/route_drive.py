#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""不带识别的完整路线：从起点格出发，按 config/route1_drive.yaml 或 route2_drive.yaml 的点走一圈，倒车回起点格。

  先起 field_bringup.launch（AMCL 定位到 field_map_20261101），关掉遥控，车摆在起点格、车头朝左，然后：
  rosrun robot_navigation route_drive.py                  # 正式跑路线 2（赛题左图，默认）
  rosrun robot_navigation route_drive.py _route:=1        # 正式跑路线 1（赛题右图）
  rosrun robot_navigation route_drive.py _dry_run:=true   # 只检查定位和打印路线，不动
  rosrun robot_navigation route_drive.py _from:=step12 _set_initial_pose:=false   # 车已在 step11 附近、定位正常时从中间接着跑

走法：点和点之间都是沿路的直线段（yaml 里每条路已经对齐到同一条中心线）。每段先原地转到线段方向，
再沿线段开：前进速度固定，偏离线段用麦轮横移拉回来，车头始终保持线段方向，所以车身走的是直线。
stop 点停下转到 look 朝向停 dwell 秒（以后在这里接识别；和当前朝向差不到 8° 就不转，免得小角度来回摆）；
via 是拐点（下一段方向差不到 8° 就不停、直接接着开）；hold: 角度 的点车头保持这个地图朝向开过去
（沿路开、同时用麦轮往一边偏）；strafe: true 的点保持车头朝向横着平移过去；end 是终点：车头转到场地 0° 倒车入库。
起步、到点按加速度 / 减速度平滑变化。
有别的节点在发 /cmd_vel（遥控空闲时也一直发 0）会拒绝启动。Ctrl+C / 被杀 / 出错时连发停车指令。
"""
import math
import os
import signal
import sys

import rosgraph
import rospy
import tf
import yaml
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from std_srvs.srv import Empty


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class Driver(object):
    def __init__(self):
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.tf = tf.TransformListener()
        self.v = float(rospy.get_param("~speed", 0.15))
        self.v_end = float(rospy.get_param("~reverse_speed", 0.08))
        self.w_max = float(rospy.get_param("~turn_speed", 0.8))
        # 提速（11-06）：长直线段（>= fast_min_len）用 fast_speed；每段出发前的拐弯转向用 corner_turn_speed。
        # 识别点上转向看目标仍用 turn_speed（转慢一点车身蹭得少、对得准）。默认不提速。
        self.v_fast = float(rospy.get_param("~fast_speed", self.v))
        self.fast_min_len = float(rospy.get_param("~fast_min_len", 0.8))
        self.w_corner = float(rospy.get_param("~corner_turn_speed", self.w_max))
        self.turn_tol = math.radians(float(rospy.get_param("~turn_tol_deg", 3.0)))
        # 原地转向：角速度 = 增益 × 剩余角度，夹在 [turn_w_min, w_max]。11-11 做成参数（原来写死 1.5 / 0.4）：
        # 快到角度时按增益减速，增益越大"收尾"越快（每次转向能省零点几秒），但冲过头的风险也越大
        self.turn_gain = float(rospy.get_param("~turn_gain", 1.5))
        self.turn_w_min = float(rospy.get_param("~turn_w_min", 0.4))   # 底盘低于 0.4 rad/s 转不动
        # 底盘低于 0.4 rad/s 转不动，几度的小角度原地转会冲过头、来回摆：偏差在这个范围内就不原地转，
        # 开动后边走边修（看点的朝向差这么点，画面也基本一样）
        self.align_tol = math.radians(float(rospy.get_param("~align_tol_deg", 8.0)))
        self.pos_tol = float(rospy.get_param("~pos_tol", 0.03))
        # 起步、刹车都按加速度慢慢变，不一下子冲出去 / 急停；末端最低速度别太低，免得最后几厘米磨蹭
        self.acc = float(rospy.get_param("~accel", 0.3))      # m/s²，起步
        self.dec = float(rospy.get_param("~decel", 0.12))     # m/s²，到点前减速
        self.v_min = float(rospy.get_param("~min_speed", 0.05))
        # 横向往线段上拉的最大速度 m/s（太大时转完向后会明显斜着走）
        self.lat_max = float(rospy.get_param("~lat_max", 0.08))
        self.v_cur = 0.0                                       # 当前沿线速度（连着走的点之间不清零）
        self.rate = rospy.Rate(20)

    def stop(self, n=5):
        self.v_cur = 0.0
        for _ in range(n):
            self.pub.publish(Twist())
            rospy.sleep(0.05)

    def pose(self):
        self.tf.waitForTransform("map", "base_footprint", rospy.Time(0), rospy.Duration(1.0))
        (x, y, _), q = self.tf.lookupTransform("map", "base_footprint", rospy.Time(0))
        return x, y, tf.transformations.euler_from_quaternion(q)[2]

    def send(self, vx, vy, wz):
        t = Twist()
        t.linear.x, t.linear.y, t.angular.z = vx, vy, wz
        self.pub.publish(t)

    def turn_to(self, yaw, timeout=15.0, w_max=None):
        w_max = self.w_max if w_max is None else w_max
        t_end = rospy.Time.now() + rospy.Duration(timeout)
        e0 = None
        while not rospy.is_shutdown() and rospy.Time.now() < t_end:
            e = wrap(yaw - self.pose()[2])
            if abs(e) < self.turn_tol:
                break
            # 冲过头（误差变号）且已经很近了就算到了，不再往回转
            if e0 is not None and e * e0 < 0 and abs(e) < 2 * self.turn_tol:
                break
            e0 = e
            w = max(self.turn_w_min, min(w_max, self.turn_gain * abs(e)))  # 太慢底盘转不动
            self.send(0, 0, math.copysign(w, e))
            self.rate.sleep()
        self.stop(2)

    def face(self, yaw, w_max=None):
        """转到 yaw；差得不多（< align_tol）就不转。"""
        if abs(wrap(yaw - self.pose()[2])) >= self.align_tol:
            self.turn_to(yaw, w_max=w_max)

    def go(self, a, b, reverse=False, body=None, slow=False, through=False, v_cap=None):
        """从 a 沿直线开到 b。车头保持线段方向（倒车时反过来），横向偏离用麦轮横移纠正。
        body：保持这个车头朝向开过去（麦轮斜着/横着走，不原地转），比如沿路开、同时往一边偏。
        slow：用慢速（倒车、纯横移）。through：到点不减速不停，直接接下一段（下一段方向差不多时用）。"""
        seg = (b[0] - a[0], b[1] - a[1])
        L = math.hypot(*seg)
        if L < 0.02:
            return
        ux, uy = seg[0] / L, seg[1] / L
        nx, ny = -uy, ux                                   # 线段左侧法向
        if body is None:
            body = wrap(math.atan2(uy, ux) + (math.pi if reverse else 0.0))
        v_set = self.v_end if (reverse or slow) else (self.v_fast if L >= self.fast_min_len else self.v)
        if v_cap is not None:                              # 这一段限速（比如从红绿灯立柱中间穿过）
            v_set = min(v_set, float(v_cap))
        self.face(body, w_max=self.w_corner)
        dt = 0.05
        t_end = rospy.Time.now() + rospy.Duration(L / v_set * 2.0 + 10.0)
        while not rospy.is_shutdown() and rospy.Time.now() < t_end:
            x, y, yaw = self.pose()
            along = (x - a[0]) * ux + (y - a[1]) * uy
            cross = (x - a[0]) * nx + (y - a[1]) * ny        # 在线段左侧为正
            left = L - along
            if left < (self.pos_tol + self.v_cur * dt if through else self.pos_tol):
                break
            # 起步按加速度爬升；到点前按减速度刹（连着走的点不刹）
            v = min(v_set, self.v_cur + self.acc * dt)
            if not through:
                v = min(v, max(self.v_min, math.sqrt(2.0 * self.dec * max(0.0, left - self.pos_tol))))
            self.v_cur = v
            lat = -max(-self.lat_max, min(self.lat_max, 1.5 * cross))       # 往线段上拉
            # 地图坐标下想要的速度，转到车体坐标
            wx, wy = v * ux + lat * nx, v * uy + lat * ny
            c, s = math.cos(yaw), math.sin(yaw)
            wz = max(-0.5, min(0.5, 2.0 * wrap(body - yaw)))
            self.send(c * wx + s * wy, -s * wx + c * wy, wz)
            self.rate.sleep()
        if not through:
            self.stop(2)


def other_cmd_vel_publishers():
    """除了自己以外还在发 /cmd_vel 的节点（遥控空闲时也一直发 0，会和路线抢底盘）。"""
    try:
        pubs, _, _ = rosgraph.Master(rospy.get_name()).getSystemState()
    except Exception:
        return []
    for topic, nodes in pubs:
        if topic == "/cmd_vel":
            return [n for n in nodes if n != rospy.get_name()]
    return []


def init_localization(start):
    """把起点格发给 AMCL，原地做几次无运动更新让粒子收拢。"""
    pub = rospy.Publisher("/initialpose", PoseWithCovarianceStamped, queue_size=1, latch=True)
    a = math.radians(start["yaw"])
    m = PoseWithCovarianceStamped()
    m.header.frame_id = "map"
    m.header.stamp = rospy.Time.now()
    m.pose.pose.position.x, m.pose.pose.position.y = start["x"], start["y"]
    m.pose.pose.orientation.z, m.pose.pose.orientation.w = math.sin(a / 2.0), math.cos(a / 2.0)
    m.pose.covariance[0] = m.pose.covariance[7] = 0.05 ** 2
    m.pose.covariance[35] = math.radians(5) ** 2
    rospy.sleep(0.5)
    pub.publish(m)
    rospy.sleep(0.5)
    try:
        rospy.wait_for_service("/request_nomotion_update", timeout=3.0)
        upd = rospy.ServiceProxy("/request_nomotion_update", Empty)
        for _ in range(30):
            upd()
            rospy.sleep(0.15)
    except rospy.ROSException:
        rospy.logwarn("没有 /request_nomotion_update（AMCL 没起？）")


def clear_stale_params():
    """rosrun 的 _xxx:= 参数会一直留在参数服务器上（上次 _dry_run:=true 这次不写也还是 true）。
    这次命令行没给的私有参数一律删掉，只认这次给的。"""
    given = {a[1:].split(":=")[0] for a in sys.argv[1:] if a.startswith("_") and ":=" in a}
    prefix = rospy.get_name() + "/"
    for name in rospy.get_param_names():
        if name.startswith(prefix) and name[len(prefix):].split("/")[0] not in given:
            try:
                rospy.delete_param(name)
            except KeyError:
                pass


def main():
    rospy.init_node("route_drive")
    clear_stale_params()
    route = str(rospy.get_param("~route", 2))
    if route not in ("1", "2"):
        rospy.logerr("_route 只能是 1 或 2")
        return
    default_cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "route%s_drive.yaml" % route)
    cfg_path = rospy.get_param("~config", default_cfg)
    rospy.loginfo("路线 %s：%s", route, os.path.basename(cfg_path))
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    start = cfg["start"]
    wps = cfg["waypoints"]
    dry = bool(rospy.get_param("~dry_run", False))
    dwell = float(rospy.get_param("~dwell", 1.0))
    look = bool(rospy.get_param("~look", True))
    from_name = str(rospy.get_param("~from", ""))
    d = Driver()

    def bye(*_):
        d.stop(10)
        rospy.signal_shutdown("stop")
        sys.exit(0)

    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, bye)
    rospy.on_shutdown(lambda: d.stop(10))
    rospy.sleep(1.0)

    others = other_cmd_vel_publishers()
    if others and not dry:
        rospy.logerr("还有节点在发 /cmd_vel：%s。先关掉遥控（Ctrl+C）再跑", ", ".join(others))
        return

    i0 = 0
    if from_name:
        names = [w["name"] for w in wps]
        if from_name not in names:
            rospy.logerr("没有叫 %s 的点，可选：%s", from_name, " ".join(names))
            return
        i0 = names.index(from_name)
    elif rospy.get_param("~set_initial_pose", True):
        init_localization(start)

    x, y, yaw = d.pose()
    rospy.loginfo("当前定位 (%.3f, %.3f) %.1f°", x, y, math.degrees(yaw))
    if not from_name:
        err_d = math.hypot(x - start["x"], y - start["y"])
        err_a = abs(wrap(yaw - math.radians(start["yaw"])))
        rospy.loginfo("起点格 (%.3f, %.3f) %.1f°：差 %.3f m、%.1f°", start["x"], start["y"], start["yaw"],
                      err_d, math.degrees(err_a))
        if err_d > 0.15 or err_a > math.radians(15):
            rospy.logerr("定位和起点格差太多：车是不是没摆在起点格、车头没朝左？（在 RViz 里看激光和墙对不对得上）")
            return
    for w in wps[i0:]:
        rospy.loginfo("  %-7s (%.3f, %.3f) %s%s", w["name"], w["x"], w["y"], w["kind"],
                      " 看 %.1f°" % w["look"] if "look" in w else "")
    if dry:
        if others:
            rospy.logwarn("还有节点在发 /cmd_vel：%s（正式跑之前要关掉）", ", ".join(others))
        return

    rospy.loginfo("3 秒后出发")
    rospy.sleep(3.0)
    prev = (x, y) if not from_name else None
    t0 = rospy.Time.now()
    run = wps[i0:]

    def heading_of(i, frm):
        """第 i 个点这一段开的时候车头朝哪（hold 指定的，否则沿线段方向）。"""
        w = run[i]
        if "hold" in w:
            return math.radians(float(w["hold"]))
        return math.atan2(w["y"] - frm[1], w["x"] - frm[0])

    try:
        for i, w in enumerate(run):
            if rospy.is_shutdown():
                break
            tgt = (float(w["x"]), float(w["y"]))
            if prev is None:            # 从中间接着跑：先从当前位置直接开过去
                px, py, _ = d.pose()
                prev = (px, py)
            # 路过的点、下一段车头方向又差不多：不停，直接接着开（不一卡一卡）
            through = (w["kind"] == "via" and i + 1 < len(run) and run[i + 1]["kind"] != "end"
                       and math.hypot(run[i + 1]["x"] - tgt[0], run[i + 1]["y"] - tgt[1]) > 0.02
                       and abs(wrap(heading_of(i + 1, tgt) - heading_of(i, prev))) < d.align_tol)
            if w["kind"] == "end":
                # 车头转到场地 0°（车头朝街道），沿直线倒进起点格
                d.turn_to(math.radians(start["yaw"]))
                d.go(prev, tgt, reverse=True)
                d.turn_to(math.radians(start["yaw"]))
            elif w.get("strafe"):
                # 保持现在的车头朝向平移过去（麦轮横着走），离线远一点再原地转
                d.go(prev, tgt, body=d.pose()[2], slow=True)
            elif "hold" in w:
                # 车头保持 hold（地图朝向）沿路开，同时麦轮往一边偏到目标点
                d.go(prev, tgt, body=math.radians(float(w["hold"])), through=through)
            else:
                d.go(prev, tgt, through=through)
            px, py, pyaw = d.pose()
            rospy.loginfo("[%5.1f s] 到 %-7s (%.3f, %.3f)，偏 %.3f m", (rospy.Time.now() - t0).to_sec(), w["name"],
                          px, py, math.hypot(px - tgt[0], py - tgt[1]))
            if w["kind"] == "stop" and look and "look" in w:
                d.face(math.radians(float(w["look"])))
                rospy.sleep(dwell)
            prev = tgt
        px, py, pyaw = d.pose()
        rospy.loginfo("跑完，用时 %.1f s；终点 (%.3f, %.3f) %.1f°，离起点格中心 %.3f m",
                      (rospy.Time.now() - t0).to_sec(), px, py, math.degrees(pyaw),
                      math.hypot(px - start["x"], py - start["y"]))
    finally:
        d.stop(10)


if __name__ == "__main__":
    main()
