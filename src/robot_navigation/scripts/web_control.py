#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""浏览器遥控 + 地图可视化 + 定点（在电脑浏览器里用，不用开 RViz）。

    http://<小车IP>:8090/

提供的接口：
  GET  /                     网页（读包里的 web/index.html）
  GET  /stream               MJPEG 视频流：地图 + 激光 + 车 + 已定点位
  GET  /snapshot.jpg         单帧（排错/截图用）
  GET  /api/state            状态 JSON（位姿、地图、频率、点位、底盘反馈）
  GET  /api/cmd?vx=&vy=&wz=  发速度（m/s, m/s, rad/s，软件限幅）
  GET  /api/stop             急停（立刻零速）
  GET  /api/teach?name=xxx   在当前位置记一个点（写进本包 map/taught_points.yaml）
  GET  /api/points           点位列表
  GET  /api/points/delete?name=xxx   删点位
  GET  /api/save_map         存一版地图（调本工作空间 tools/map_save.sh --keep，后台跑）

安全：
  · /cmd_vel 由 chassis_bridge 消费；这个节点只发指令，不碰串口
  · 网页 0.5 秒不刷新就自动零速；chassis_bridge 里还有一层 0.5 秒超时
"""
from __future__ import print_function

import io
import json
import math
import os
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import rospkg
import rospy
import tf
import yaml
from teaching_capture import (OPERATION_LOCK, capture_heading, capture_point, point_headings, rebuild_point,
                              validate_points, write_points)
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from point_navigation import PointNavigator
from nav_msgs.msg import OccupancyGrid, Odometry
from PIL import Image, ImageDraw, ImageFont
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty

# ------------------------------------------------------------------ 画面参数
VIEW_W, VIEW_H = 760, 620
VIEW_MPP = 0.015                      # 米/像素（约 11.4 m × 9.3 m 视野）
BG = (24, 24, 28)
GRID = (44, 46, 54)
FREE = (214, 214, 214)
OCC = (18, 18, 18)
UNKNOWN = (58, 60, 68)
LASER = (255, 72, 72)
ROBOT = (255, 176, 40)
POINT = (80, 230, 140)
TEXT = (230, 230, 235)

# 存图走本工作空间的 tools/map_save.sh：本包在 <工作空间>/src/robot_navigation 下，往上两级就是工作空间根
PACKAGE_DIR = rospkg.RosPack().get_path("robot_navigation")
POINTS_FILE = os.path.join(PACKAGE_DIR, "map", "taught_points.yaml")
POINTS_DETAILS_FILE = os.path.join(PACKAGE_DIR, "map", "taught_points.json")
SAVE_MAP_SH = os.path.join(os.path.dirname(os.path.dirname(PACKAGE_DIR)), "tools", "map_save.sh")

# 点位名可能是中文：默认位图字体只能画拉丁字符，画中文会抛异常（整帧渲染失败）。
# 这里按顺序找一个 CJK 字体，找不到就退回默认字体 + 画字时兜底（异常只跳过标签，不毁整帧）。
FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
]
LABEL_FONT = None
for _p in FONT_CANDIDATES:
    if os.path.exists(_p):
        try:
            LABEL_FONT = ImageFont.truetype(_p, 14)
            break
        except Exception:
            LABEL_FONT = None


def load_points():
    with OPERATION_LOCK:
        return read_points()


def light_point(point):
    """状态接口只发坐标与朝向概要，激光点每帧都发会把轮询放大几十倍。"""
    headings = point.get('headings') or [point]
    return {'name': point['name'], 'x': point['x'], 'y': point['y'],
            'yaw': point.get('yaw_deg', point.get('yaw', 0.0)), 'time': point.get('time', ''),
            'headings': [{'index': index, 'yaw': heading.get('yaw_deg', 0.0),
                          'note': heading.get('note', ''),
                          'capture_time': heading.get('capture_time', ''),
                          'sample_count': heading.get('sample_count', 0),
                          'position_spread_mm': round(heading.get('position_spread_mm', 0.0), 2),
                          'yaw_spread_deg': round(heading.get('yaw_spread_deg', 0.0), 3),
                          'validated': heading.get('validated', False)}
                         for index, heading in enumerate(headings)]}


def read_points():
    if not os.path.exists(POINTS_FILE):
        return []
    with open(POINTS_FILE, encoding='utf-8') as source:
        coordinates = yaml.safe_load(source) or {}
    details = {}
    if os.path.exists(POINTS_DETAILS_FILE):
        with open(POINTS_DETAILS_FILE, encoding='utf-8') as source:
            details = {point['name']: point for point in json.load(source)['points']}
    pts = []
    for name, xy in coordinates.items():
        item = details.get(name, {})
        if item and (abs(item['x'] - xy[0]) > 0.000001 or abs(item['y'] - xy[1]) > 0.000001):
            raise ValueError('点位文件内容不一致：' + name)
        pts.append({**item, 'name': name, 'x': xy[0], 'y': xy[1],
                    'yaw': item.get('yaw_deg', item.get('yaw', 0.0)), 'time': item.get('time', '')})
    return pts


class Monitor(object):
    """收 ROS 数据 + 缓存渲染结果。"""

    def __init__(self):
        self.lock = threading.RLock()
        self.map_msg = None
        self.map_img = None
        self.map_key = None
        self.scan = None
        self.scan_stamp = 0.0
        self.scan_hz = 0.0
        self.vel = None
        self.vel_stamp = 0.0
        self.odom_stamp = 0.0
        self.odom_hz = 0.0
        self.cmd = (0.0, 0.0, 0.0)
        self.cmd_stamp = 0.0
        self.amcl_pose = None
        self.tf_listener = tf.TransformListener()
        rospy.Subscriber("/map", OccupancyGrid, self.on_map, queue_size=1)
        rospy.Subscriber("/scan", LaserScan, self.on_scan, queue_size=1)
        rospy.Subscriber("/vel_raw", Twist, self.on_vel, queue_size=5)
        rospy.Subscriber("/odom_raw", Odometry, self.on_odom, queue_size=5)
        rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.on_amcl, queue_size=1)
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self._mode_cache = (0.0, "none")
        self.navigation = PointNavigator(self)
        self.idle_update_at = 0.0
        self.idle_update_count = 8
        self.idle_update = rospy.ServiceProxy("/request_nomotion_update", Empty)
        rospy.Subscriber("/initialpose", PoseWithCovarianceStamped, self.on_initial_pose, queue_size=1)

    def on_initial_pose(self, message):
        with self.lock:
            self.idle_update_count = 0

    def on_amcl(self, message):
        with self.lock:
            self.amcl_pose = message

    def update_idle_localization(self, allow_navigation=False):
        now = time.monotonic()
        if not allow_navigation and self.navigation.state()["active"]:
            return
        with self.lock:
            if now - self.idle_update_at < 1.0 or self.idle_update_count >= 8:
                return
            stationary = (self.vel is not None and time.time() - self.vel_stamp < 0.5
                          and max(abs(value) for value in self.vel) < 0.005
                          and max(abs(value) for value in self.cmd) < 0.001)
            covariance = self.amcl_pose.pose.covariance if self.amcl_pose is not None else None
            if covariance is not None and max(covariance[0], covariance[7]) <= 0.02 ** 2 and covariance[35] <= math.radians(0.5) ** 2:
                return
            if not stationary or self.mode() != "localize":
                return
            self.idle_update_at = now
            self.idle_update_count += 1
        self.idle_update()

    # -------------------------------------------------------------- 回调
    def on_map(self, msg):
        with self.lock:
            self.map_msg = msg

    def on_scan(self, msg):
        with self.lock:
            now = time.time()
            if self.scan_stamp:
                dt = now - self.scan_stamp
                if 0.0 < dt < 1.0:
                    hz = 1.0 / dt
                    self.scan_hz = hz if not self.scan_hz else 0.85 * self.scan_hz + 0.15 * hz
            self.scan = msg
            self.scan_stamp = now

    def on_odom(self, msg):
        with self.lock:
            now = time.time()
            if self.odom_stamp:
                dt = now - self.odom_stamp
                if 0.0 < dt < 1.0:
                    hz = 1.0 / dt
                    self.odom_hz = hz if not self.odom_hz else 0.85 * self.odom_hz + 0.15 * hz
            self.odom_stamp = now

    def on_vel(self, msg):
        with self.lock:
            self.vel = (msg.linear.x, msg.linear.y, msg.angular.z)
            self.vel_stamp = time.time()

    # -------------------------------------------------------------- 模式
    def mode(self):
        """mapping 支持建图与定点；localize 使用 AMCL 定位。"""
        now = time.time()
        if now - self._mode_cache[0] < 1.0:
            return self._mode_cache[1]
        # 用进程判断（master 里被 kill -9 的节点会留注册，不可靠）
        def running(pat):
            try:
                return subprocess.call(["pgrep", "-f", pat],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
            except Exception:
                return False
        if running("slam_gmapping"):
            m = "mapping"
        elif running("amcl"):
            m = "localize"
        else:
            m = "none"
        self._mode_cache = (now, m)
        return m

    def converge(self, n=20):
        """AMCL 才有 /request_nomotion_update：连调 n 次让粒子收拢（= field_teach.py 的做法）。"""
        rospy.wait_for_service("/request_nomotion_update", timeout=2.0)
        upd = rospy.ServiceProxy("/request_nomotion_update", Empty)
        for _ in range(n):
            upd()
            rospy.sleep(0.15)
        return True

    def relocalize(self):
        """AMCL 的 /global_localization：把粒子撒满全图，靠激光重新收敛。"""
        try:
            rospy.wait_for_service("/global_localization", timeout=3.0)
            rospy.ServiceProxy("/global_localization", Empty)()
            return True
        except Exception:
            return False

    def release_cmd(self):
        """取消 /cmd_vel 发布者：route_drive.py 会拒绝「还有别人在发 /cmd_vel」，跑路线前要先放掉。"""
        try:
            if self.cmd_pub is not None:
                self.cmd_pub.unregister()
        except Exception:
            pass
        self.cmd_pub = None

    # -------------------------------------------------------------- 指令
    def send_cmd(self, vx, vy, wz):
        vx = max(-0.15, min(0.15, vx))
        vy = max(-0.15, min(0.15, vy))
        wz = max(-1.2, min(1.2, wz))
        if self.cmd_pub is None:
            self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
            rospy.sleep(0.2)
        t = Twist()
        t.linear.x, t.linear.y, t.angular.z = vx, vy, wz
        self.cmd_pub.publish(t)
        with self.lock:
            self.cmd = (vx, vy, wz)
            self.cmd_stamp = time.time()

    def stop(self):
        self.send_cmd(0.0, 0.0, 0.0)

    # -------------------------------------------------------------- 位姿
    def pose(self):
        try:
            (t, q) = self.tf_listener.lookupTransform("map", "base_footprint", rospy.Time(0))
            yaw = math.atan2(2 * (q[3] * q[2] + q[0] * q[1]), 1 - 2 * (q[1] * q[1] + q[2] * q[2]))
            return (t[0], t[1], yaw)
        except Exception:
            return None

    def laser_in_map(self):
        """返回 (x0,y0,yaw) —— laser_link 在 map 里的位姿。"""
        try:
            (t, q) = self.tf_listener.lookupTransform("map", "laser_link", rospy.Time(0))
            yaw = math.atan2(2 * (q[3] * q[2] + q[0] * q[1]), 1 - 2 * (q[1] * q[1] + q[2] * q[2]))
            return (t[0], t[1], yaw)
        except Exception:
            return None

    # -------------------------------------------------------------- 状态
    def state(self):
        with self.lock:
            scan_hz = self.scan_hz
            vel = self.vel
            vel_age = (time.time() - self.vel_stamp) if self.vel_stamp else -1
            cmd = self.cmd
            mm = self.map_msg
        pose = self.pose()
        st = {
            "ok": True,
            "pose": None if pose is None else {"x": round(pose[0], 3), "y": round(pose[1], 3),
                                               "yaw": round(math.degrees(pose[2]), 1)},
            "mode": self.mode(),
            "scan_hz": round(scan_hz, 1),
            "odom_hz": round(self.odom_hz, 1),
            "vel": None if vel is None else {"vx": round(vel[0], 3), "vy": round(vel[1], 3),
                                             "wz": round(vel[2], 3), "age": round(vel_age, 2)},
            "cmd": {"vx": cmd[0], "vy": cmd[1], "wz": cmd[2]},
            "map": None if mm is None else {"w": mm.info.width, "h": mm.info.height,
                                            "res": round(mm.info.resolution, 3)},
            "points": [light_point(point) for point in load_points()],
            "points_file": POINTS_FILE,
            "navigation": self.navigation.state(),
            "time": time.strftime("%H:%M:%S"),
        }
        return st

    # -------------------------------------------------------------- 渲染
    def _map_image(self):
        """把 /map 转成 PIL 图（缓存，地图变了才重建）。"""
        with self.lock:
            mm = self.map_msg
            if mm is None:
                return None, None
            key = (mm.info.width, mm.info.height, round(mm.info.resolution, 4),
                   round(mm.info.origin.position.x, 3), round(mm.info.origin.position.y, 3),
                   mm.header.stamp.to_sec(), mm.info.width * mm.info.height)
            if self.map_img is not None and self.map_key == key:
                return self.map_img, mm
            w, h, res = mm.info.width, mm.info.height, mm.info.resolution
            arr = np.array(mm.data, dtype=np.int16).reshape(h, w)
            rgb = np.zeros((h, w, 3), dtype=np.uint8)
            rgb[:] = UNKNOWN
            rgb[arr == 0] = FREE
            rgb[arr > 50] = OCC
            rgb[(arr > 0) & (arr <= 50)] = FREE
            img = Image.fromarray(np.flipud(rgb), "RGB")
            k = res / VIEW_MPP                            # 每个格子几个像素
            img = img.resize((max(1, int(round(w * k))), max(1, int(round(h * k)))), Image.NEAREST)
            self.map_img = img
            self.map_key = key
            return img, mm

    def render(self):
        try:
            return self._render()
        except Exception as ex:
            import traceback
            traceback.print_exc()
            try:
                img = Image.new("RGB", (VIEW_W, VIEW_H), BG)
                d = ImageDraw.Draw(img)
                msg = "render error: %s" % ex
                try:
                    f = LABEL_FONT or None
                    d.text((12, 12), msg[:110], fill=(255, 90, 90), font=f)
                except Exception:
                    d.text((12, 12), "render error", fill=(255, 90, 90))
                buf = io.BytesIO()
                img.save(buf, "JPEG", quality=70)
                return buf.getvalue()
            except Exception:
                return b""

    def _render(self):
        canvas = Image.new("RGB", (VIEW_W, VIEW_H), BG)
        draw = ImageDraw.Draw(canvas)
        map_img, mm = self._map_image()
        pose = self.pose()
        if pose is None:
            pose = (0.0, 0.0, 0.0)
        cx, cy = pose[0], pose[1]

        def w2s(wx, wy):
            return (int(round((wx - cx) / VIEW_MPP + VIEW_W / 2.0)),
                    int(round(-(wy - cy) / VIEW_MPP + VIEW_H / 2.0)))

        # 地图
        if map_img is not None:
            ox = mm.info.origin.position.x
            oy = mm.info.origin.position.y
            left, top = w2s(ox, oy + mm.info.height * mm.info.resolution)
            canvas.paste(map_img, (left, top))

        # 1 米网格
        x_lo, x_hi = cx - VIEW_W * VIEW_MPP / 2, cx + VIEW_W * VIEW_MPP / 2
        y_lo, y_hi = cy - VIEW_H * VIEW_MPP / 2, cy + VIEW_H * VIEW_MPP / 2
        gx = math.floor(x_lo)
        while gx <= x_hi:
            px, _ = w2s(gx, cy)
            draw.line([(px, 0), (px, VIEW_H)], fill=GRID)
            gx += 1.0
        gy = math.floor(y_lo)
        while gy <= y_hi:
            _, py = w2s(cx, gy)
            draw.line([(0, py), (VIEW_W, py)], fill=GRID)
            gy += 1.0

        # 激光
        with self.lock:
            scan = self.scan
        if scan is not None:
            src = self.laser_in_map() or (pose[0], pose[1], pose[2] + math.pi)
            lx, ly, lyaw = src
            pts = []
            for i, r in enumerate(scan.ranges):
                if not (0.10 < r < 25.0):
                    continue
                a = scan.angle_min + i * scan.angle_increment + lyaw
                x, y = lx + r * math.cos(a), ly + r * math.sin(a)
                pts.append(w2s(x, y))
            if pts:
                draw.point(pts, fill=LASER)
                draw.point([(x + 1, y) for (x, y) in pts], fill=LASER)

        navigation = self.navigation.state()
        if len(navigation["path"]) > 1:
            draw.line([w2s(x, y) for x, y in navigation["path"]], fill=(70, 160, 255), width=3)

        # 已定点位
        for p in load_points():
            px, py = w2s(p["x"], p["y"])
            if -20 < px < VIEW_W + 20 and -20 < py < VIEW_H + 20:
                draw.ellipse([px - 4, py - 4, px + 4, py + 4], outline=POINT, width=2)
                heading = math.radians(p["yaw"])
                tip = (px + 22 * math.cos(heading), py - 22 * math.sin(heading))
                draw.line([(px, py), tip], fill=POINT, width=2)
                for side in (-0.5, 0.5):
                    tail = (tip[0] - 8 * math.cos(heading + side), tip[1] + 8 * math.sin(heading + side))
                    draw.line([tail, tip], fill=POINT, width=2)
                try:
                    if LABEL_FONT is not None:
                        draw.text((px + 6, py - 6), p["name"], fill=POINT, font=LABEL_FONT)
                    else:
                        draw.text((px + 6, py - 6), p["name"], fill=POINT)
                except Exception:
                    pass   # 画不出字就算了，别毁整帧

        # 车（橙色箭头）
        px, py = w2s(pose[0], pose[1])
        yaw = pose[2]
        L = 0.30 / VIEW_MPP
        Wd = 0.11 / VIEW_MPP
        ca, sa = math.cos(yaw), math.sin(yaw)

        def rot(dx, dy):
            # 车体系偏移 (前, 左) → 屏幕像素（屏幕 y 轴朝下）
            sx = dx * ca - dy * sa
            sy = -(dx * sa + dy * ca)
            return (px + sx, py + sy)

        nose = rot(L, 0)
        left = rot(-L * 0.4, Wd)
        right = rot(-L * 0.4, -Wd)
        draw.polygon([nose, left, right], fill=ROBOT)
        draw.line([rot(-L * 0.4, 0), rot(L * 1.4, 0)], fill=ROBOT, width=1)

        # 比例尺
        bar = int(round(1.0 / VIEW_MPP))
        draw.line([(16, VIEW_H - 20), (16 + bar, VIEW_H - 20)], fill=TEXT, width=2)
        draw.text((16, VIEW_H - 36), "1 m", fill=TEXT)

        if map_img is None:
            draw.text((VIEW_W / 2 - 60, 20), "waiting for /map ...", fill=TEXT)

        buf = io.BytesIO()
        canvas.save(buf, "JPEG", quality=75)
        return buf.getvalue()


# ------------------------------------------------------------------ HTTP
class Handler(BaseHTTPRequestHandler):
    monitor = None
    web_dir = None
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _send(self, code, ctype, body, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj):
        self._send(200, "application/json", json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def do_GET(self):
        try:
            return self.dispatch_get()
        except (ValueError, rospy.ROSException, tf.Exception, subprocess.SubprocessError) as error:
            return self._json({'ok': False, 'err': str(error)})

    def dispatch_get(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        path = u.path

        if path == '/api/evidence':
            root = os.path.realpath(os.path.expanduser('~/final_ws/data/record'))
            requested = os.path.realpath(os.path.expanduser(q.get('path', [''])[0]))
            if os.path.commonpath([root, requested]) != root or not os.path.isfile(requested):
                return self._json({'ok': False, 'err': '证据图片不存在或不在运行数据目录'})
            if os.path.splitext(requested)[1].lower() not in ('.jpg', '.jpeg', '.png'):
                return self._json({'ok': False, 'err': '仅支持证据图片'})
            with Image.open(requested) as image:
                output = io.BytesIO()
                image.convert('RGB').save(output, format='JPEG', quality=90)
            return self._send(200, 'image/jpeg', output.getvalue())

        if path in ("/", "/index.html"):
            fn = os.path.join(self.web_dir, "index.html")
            try:
                with io.open(fn, "rb") as f:
                    body = f.read()
            except Exception as ex:
                body = ("<h3>找不到 %s（%s）</h3>" % (fn, ex)).encode("utf-8")
            return self._send(200, "text/html; charset=utf-8", body)

        if path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            period = 1.0 / float(rospy.get_param("~fps", 8.0))
            try:
                while not rospy.is_shutdown():
                    t0 = time.time()
                    jpg = self.monitor.render()
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                    dt = time.time() - t0
                    if dt < period:
                        time.sleep(period - dt)
            except OSError:
                pass
            return

        if path == "/snapshot.jpg":
            return self._send(200, "image/jpeg", self.monitor.render())

        if path == "/api/state":
            return self._json(self.monitor.state())

        if path == "/api/trash_preview.jpg":
            with self.monitor.navigation.lock:
                preview = self.monitor.navigation.trash_preview
            if preview is None:
                return self._send(404, "text/plain; charset=utf-8", "等待垃圾桶识别".encode("utf-8"))
            return self._send(200, "image/jpeg", preview)

        if path.startswith("/api/navigation/"):
            try:
                operation = path.rsplit("/", 1)[-1]
                if operation == "start":
                    name = q.get("name", [""])[0]
                    point = next((point for point in load_points() if point["name"] == name), None)
                    if point is None:
                        raise ValueError("请选择已经记录的位置")
                    heading = float(q['heading'][0]) if 'heading' in q else None
                    cursor = int(q['cursor'][0]) if 'cursor' in q else None
                    navigation = self.monitor.navigation.start(point, heading, q.get('from', [None])[0], cursor)
                elif operation == 'rotate':
                    navigation = self.monitor.navigation.start_rotation(float(q['degrees'][0]))
                elif operation == "route":
                    navigation = self.monitor.navigation.start_route(q.get("route", ["auto"])[0])
                elif operation == "resume":
                    resume = json.loads(q["resume"][0])
                    navigation = self.monitor.navigation.start_route(q["route"][0], resume)
                elif operation == "route_check":
                    config, points = self.monitor.navigation.prepare_route(q.get("route", ["auto"])[0])
                    return self._json({"ok": True, "msg": "路线点位、起始位置、朝向、定位与雷达检查通过", "points": len(points),
                                       "dwell": config["task_dwell"], "alignment": self.monitor.navigation.last_alignment,
                                       "localization": self.monitor.navigation.last_quality})
                elif operation == "localization":
                    self.monitor.navigation.check_sensors()
                    quality = self.monitor.navigation.alignment_details()
                    with self.monitor.navigation.lock:
                        self.monitor.navigation.last_quality = quality
                    return self._json({"ok": True, "localization": quality})
                elif operation == "stop":
                    self.monitor.navigation.stop()
                    navigation = self.monitor.navigation.state()
                elif operation == "heartbeat":
                    self.monitor.navigation.heartbeat()
                    navigation = self.monitor.navigation.state()
                else:
                    raise ValueError("未知的单点导航操作")
                return self._json({"ok": True, "navigation": navigation, "msg": navigation["message"]})
            except (ValueError, RuntimeError) as error:
                return self._json({"ok": False, "err": str(error)})

        if path == "/api/cmd":
            self.monitor.navigation.stop()
            def f(k, d=0.0):
                try:
                    return float(q.get(k, [d])[0])
                except Exception:
                    return d
            self.monitor.send_cmd(f("vx"), f("vy"), f("wz"))
            return self._json({"ok": True, "cmd": self.monitor.state()["cmd"]})

        if path == "/api/relocalize":
            self.monitor.navigation.stop()
            ok = self.monitor.relocalize()
            return self._json({"ok": ok,
                               "msg": "粒子已撒满全图，请慢慢开 1~2 米或原地转一圈让它收拢" if ok
                                      else "现在不是定位模式（AMCL 没在跑），没法重定位"})

        if path == "/api/release":
            self.monitor.navigation.stop()
            self.monitor.stop()
            self.monitor.release_cmd()
            return self._json({"ok": True, "msg": "/cmd_vel 发布者已释放（现在可以跑 route_drive）"})

        if path == "/api/stop":
            self.monitor.navigation.stop()
            self.monitor.stop()
            return self._json({"ok": True})

        if path == "/api/teach":
            if self.monitor.navigation.state()["active"]:
                return self._json({"ok": False, "err": "请停止单点导航后记录点位"})
            name = (q.get("name", [""])[0] or "").strip()
            if not name:
                return self._json({"ok": False, "err": "名字不能为空"})
            create = (q.get("create", ["0"])[0] or "0").strip() == "1"
            mode = self.monitor.mode()
            if mode not in ("mapping", "localize"):
                return self._json({"ok": False, "err": "请启动建图或定位后记录点位"})
            if create and any(item['name'] == name for item in load_points()):
                return self._json({"ok": False, "err": "点位名 %s 已经存在，换一个名字，或者用同名覆盖记录" % name})
            self.monitor.stop()
            with OPERATION_LOCK:
                point = capture_point(self.monitor, name, mode)
                points = [item for item in load_points() if item['name'] != name]
                points.append(point)
                write_points(POINTS_FILE, POINTS_DETAILS_FILE, points)
            message = ('已新建点位 %s：位置 %.6f, %.6f 米，朝向 %.3f 度；16份数据位置变化 %.2f 毫米、朝向变化 %.3f 度。' if create
                       else '已覆盖记录 %s：位置 %.6f, %.6f 米，朝向 %.3f 度；16份数据位置变化 %.2f 毫米、朝向变化 %.3f 度。') % (
                name, point['x'], point['y'], point['yaw'], point['position_spread_mm'], point['yaw_spread_deg'])
            return self._json({'ok': True, 'msg': message, 'mode': mode,
                               'point': light_point(point), 'points': [light_point(item) for item in points]})

        if path == "/api/teach/heading":
            if self.monitor.navigation.state()["active"]:
                return self._json({"ok": False, "err": "请停止单点导航后记录朝向"})
            name = (q.get("name", [""])[0] or "").strip()
            note = (q.get("note", [""])[0] or "").strip()
            if not name:
                return self._json({"ok": False, "err": "名字不能为空"})
            mode = self.monitor.mode()
            if mode not in ("mapping", "localize"):
                return self._json({"ok": False, "err": "请启动建图或定位后记录朝向"})
            points = load_points()
            current = [item for item in points if item['name'] == name]
            if not current:
                return self._json({"ok": False, "err": "没有这个点位，请先用记录点位"})
            self.monitor.stop()
            with OPERATION_LOCK:
                updated = capture_heading(self.monitor, current[0], mode, note)
                write_points(POINTS_FILE, POINTS_DETAILS_FILE,
                             [updated if item['name'] == name else item for item in points])
            heading = updated['headings'][-1]
            message = '已为 %s 记录第 %d 个朝向：%.3f 度；16份数据位置变化 %.2f 毫米、朝向变化 %.3f 度。%s' % (
                name, len(updated['headings']), heading['yaw_deg'], heading['position_spread_mm'],
                heading['yaw_spread_deg'], note)
            return self._json({'ok': True, 'msg': message, 'mode': mode, 'point': light_point(updated)})

        if path == "/api/headings/delete":
            if self.monitor.navigation.state()["active"]:
                return self._json({"ok": False, "err": "请停止单点导航后删除朝向"})
            name = (q.get("name", [""])[0] or "").strip()
            try:
                index = int(q.get("index", [""])[0])
            except ValueError:
                return self._json({"ok": False, "err": "朝向序号无效"})
            with OPERATION_LOCK:
                points = load_points()
                current = [item for item in points if item['name'] == name]
                if not current:
                    return self._json({"ok": False, "err": "没有这个点位"})
                headings = [dict(heading) for heading in point_headings(current[0])]
                if not 0 <= index < len(headings):
                    return self._json({"ok": False, "err": "朝向序号超出范围"})
                if len(headings) <= 1:
                    return self._json({"ok": False, "err": "点位至少保留一个朝向，删除整个点位请用删除按钮"})
                del headings[index]
                updated = rebuild_point(current[0], headings)
                write_points(POINTS_FILE, POINTS_DETAILS_FILE,
                             [updated if item['name'] == name else item for item in points])
            return self._json({"ok": True, "msg": "已删除 %s 的第 %d 个朝向" % (name, index + 1),
                               "point": light_point(updated)})

        if path == "/api/points":
            return self._json({"ok": True, "points": load_points(), "file": POINTS_FILE})

        if path == "/api/points/delete":
            if self.monitor.navigation.state()["active"]:
                return self._json({"ok": False, "err": "请停止单点导航后删除点位"})
            name = (q.get("name", [""])[0] or "").strip()
            if not name or not os.path.exists(POINTS_FILE):
                return self._json({"ok": False, "err": "没有这个点"})
            with OPERATION_LOCK:
                points = load_points()
                removed = sum(point['name'] == name for point in points)
                write_points(POINTS_FILE, POINTS_DETAILS_FILE, [point for point in points if point['name'] != name])
            return self._json({"ok": removed > 0, "removed": removed, "points": load_points()})

        if path == "/api/save_map":
            if self.monitor.navigation.state()["active"]:
                return self._json({"ok": False, "err": "请停止单点导航后保存"})
            self.monitor.stop()
            with OPERATION_LOCK:
                with self.monitor.lock:
                    grid = self.monitor.map_msg
                if grid is None:
                    raise ValueError('当前没有地图，无法保存')
                points = validate_points(grid, load_points())
                write_points(POINTS_FILE, POINTS_DETAILS_FILE, points)
                completed = subprocess.run(['bash', SAVE_MAP_SH, '--keep'], check=True,
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           universal_newlines=True, timeout=75)
            return self._json({'ok': True, 'msg': completed.stdout})

        return self._send(404, "text/plain; charset=utf-8", "not found".encode("utf-8"))


def main():
    rospy.init_node("web_control")
    port = int(rospy.get_param("~port", 8090))
    web_dir = rospy.get_param("~web_dir", "")
    if not web_dir:
        try:
            import rospkg
            web_dir = os.path.join(rospkg.RosPack().get_path("robot_navigation"), "web")
        except Exception:
            web_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web")
    Handler.monitor = Monitor()
    Handler.web_dir = web_dir
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, name="http", daemon=True).start()
    rospy.loginfo("web_control: 浏览器打开 http://<小车IP>:%d/  （网页目录 %s）", port, web_dir)
    rospy.on_shutdown(srv.shutdown)
    rospy.on_shutdown(Handler.monitor.navigation.stop)
    while not rospy.is_shutdown():
        Handler.monitor.update_idle_localization()
        rospy.sleep(0.5)


if __name__ == "__main__":
    main()
