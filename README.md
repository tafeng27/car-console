# 车端网页控制台与到点停车

一台四轮麦轮实车（ROS Noetic）上的网页控制台。工控机起服务，浏览器打开 `http://<车IP>:8090/` 就能遥控、看地图、记录点位、跑规定路线。仓库只放代码，地图、点位、模型权重和现场数据都不在里面。

## 目录

| 路径 | 作用 |
|---|---|
| `src/robot_navigation/scripts/web_control.py` | 控制台服务：MJPEG 地图画面、遥控、点位记录、地图保存、导航接口 |
| `src/robot_navigation/web/index.html` | 控制台网页 |
| `src/robot_navigation/scripts/teaching_capture.py` | 点位与朝向的记录、地图一致性校验、点位文件写入 |
| `src/robot_navigation/scripts/point_navigation.py` | 单点导航：按规定路线取连续动作、车身通行范围检查 |
| `src/robot_navigation/scripts/navigation_trial.py` | 路线执行：行进、转向、识别停车、到点复核 |
| `src/robot_navigation/scripts/precision_navigation.py` | 记录激光匹配、地图投影、静止复核、微调脉冲 |
| `src/robot_navigation/scripts/navigation_geometry.py` | 点位在最终地图中的投影、人工停车校正 |
| `src/robot_navigation/scripts/navigation_plan.py` | 路线配置加载与校验 |
| `src/robot_navigation/scripts/route_navigation.py` | 从路线里截取从当前点到目标的连续动作 |
| `src/robot_navigation/scripts/route_drive.py` | 底盘直线段驱动、原地转向、加速度与减速度 |
| `src/robot_navigation/scripts/localization_quality.py` | 墙面贴合率与全部激光匹配率 |
| `src/robot_navigation/config/navigation_trial.yaml` | 路线、点位文件、精确停车参数、限速 |
| `tools/` | 建图与定位启动脚本、存图、静止位姿复核 |

## 运行环境

- Ubuntu 18.04 + ROS Noetic，Python 3.8，只跑 CPU
- Python 依赖：`numpy`、`opencv-python`、`scipy`、`Pillow`、`PyYAML`、`rospkg`
- 底盘、雷达、IMU、相机由 `real_robot_bringup`、`car_bringup`、`leishen`、`yesense_imu`、`OrbbecSDK_ROS` 等包提供；本仓库不含驱动
- 话题固定为 `/cmd_vel`、`/odom`、`/scan`、`/map`、`/amcl_pose`，坐标系固定为 `map → odom → base_footprint → base_link → laser_link`

## 启动

```bash
bash tools/map_start.sh           # 建图 + 网页，重开会话会把旧地图与旧点位备份后清空
bash tools/navigation_ready.sh    # 定位 + 网页，按配置里的起停点初始化 AMCL
```

两者都通过 `tools/session_common.sh` 启动并记录进程号，`stop_session` 会发零速再结束进程。也可以单独起：

```bash
python3 src/robot_navigation/scripts/web_control.py
```

前提是 `/map`、`/scan`、`/odom` 与 `map → base_footprint` 变换已经在发布。

## 网页接口

| 接口 | 作用 |
|---|---|
| `/api/state` | 位姿、雷达频率、底盘反馈、点位概要、导航状态 |
| `/api/cmd?vx=&vy=&wz=` | 发速度，软件限幅 |
| `/api/stop` | 零速 |
| `/api/teach?name=&create=1` | 在当前位置记录点位（位置与朝向一起记）；`create=1` 时同名会被拒绝，用于新建点位 |
| `/api/teach/heading?name=&note=` | 给已有点位追加一个朝向 |
| `/api/headings/delete?name=&index=` | 删除某个朝向 |
| `/api/points`、`/api/points/delete?name=` | 点位列表与删除 |
| `/api/save_map` | 校验全部点位后保存一版地图 |
| `/api/relocalize`、`/api/release` | 重新定位、释放 `/cmd_vel` 发布者 |
| `/api/navigation/{point,rotate,route,resume,route_check,localization,stop,heartbeat}` | 单点导航、路线、检查与心跳 |

## 点位与朝向的数据结构

一个点位可以有多个朝向，用来覆盖同一个位置上的不同拍摄角度。`map/taught_points.json`：

```json
{
  "frame_id": "map",
  "session": "20261118_131728_602073",
  "points": [
    {
      "name": "社区3",
      "x": 3.3688, "y": 1.8587, "yaw_deg": 179.83, "yaw_rad": 3.1387,
      "scan_base": [[...]],
      "laser_extrinsic": [0.1451, -0.0025, 3.14159265],
      "capture_time": "2026-11-18T13:02:22",
      "position_spread_mm": 3.02, "yaw_spread_deg": 0.09,
      "validated": true,
      "headings": [
        {"x": 3.3688, "y": 1.8587, "yaw_deg": 179.83, "scan_base": [[...]], "note": "", "validated": true},
        {"x": 3.3701, "y": 1.8590, "yaw_deg": 90.10, "scan_base": [[...]], "note": "正对仪表", "validated": true}
      ]
    }
  ]
}
```

- `headings` 里每一项是一次独立记录：自己的位姿、激光点、采样离散和备注
- 顶层字段复制第一个朝向，旧代码按单朝向读取不会坏
- 点位坐标 `x`、`y` 取所有朝向的均值
- 到点复核时会把该点位的每个朝向各匹配一次，取内点比例最高的那份
- 网页上“新建点位”遇到同名会拒绝，“同名覆盖记录”才会改写已有点位；`session` 字段记录点位所属的建图会话，写入时自动沿用

`map/taught_points.yaml` 只有 `名字: [x, y]`，供其他程序快速读取坐标；两份文件由 `teaching_capture.write_points` 同时写入。

## 到点停车的做法

1. 记录点位时连续取 16 份激光与位姿，要求位置变化不超过 5 毫米、朝向变化不超过 0.5 度，再写入文件。
2. 保存地图时用最终地图复核每个朝向的记录激光，墙面贴合率不低于 95%、全部激光匹配率不低于 85% 才算通过。
3. 导航目标是记录激光在最终地图里拟合出的位姿，`map/arrival_calibrations.json` 里再叠加人工测量过的停车校正。
4. 到点后停车，连续取 4 份激光与记录激光做相对匹配，得到“离记录位置多远”的估计，与地图独立投影互相复核，超过阈值就停车报错。
5. 到点容差默认 15 毫米，超差时按车体坐标单轴发一次微调脉冲；脉冲按速度提前收手，补掉里程计滞后。
6. 直线段末端按 `limits.stop_latency` 预留刹车余量，普通移动在离目标 6 毫米处发零速。

## 已知限制

- 地图、点位、AMCL 三者必须用同一套雷达安装参数，否则同一物理位置在不同坐标下的读数会差几厘米
- 对称场地里 AMCL 会锁到错误位置，交叉路口这类地方需要靠记录激光复核
- 地图是 5 厘米一格，毫米级停车靠记录激光的相对匹配，地图本身给不到这个精度
- 本仓库不含地图、点位、模型权重、人工停车校正与现场数据
