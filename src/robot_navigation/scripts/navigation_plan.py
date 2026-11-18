import argparse
import json
import math
from pathlib import Path

import yaml

MODES = {"forward", "reverse", "left", "right"}
TASKS = {"traffic_light_1", "traffic_light_2", "crowd_A", "crowd_B", "building_A", "building_B",
         "building_C", "building_D", "plate_1", "plate_2", "plate_3", "trash", "meter", "ebike", "prohibited_sign"}


def load_plan(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    point_path = (path.parent / config["points"]).resolve()
    document = json.loads(point_path.read_text(encoding="utf-8"))
    if document["frame_id"] != "map":
        raise ValueError("导航点位需要 map 坐标系")
    points = {point["name"]: {**point, 'recording_session': document['session']} for point in document["points"]}
    if len(points) != len(document["points"]):
        raise ValueError("点位名称重复")
    for point in points.values():
        if not all(math.isfinite(point[key]) for key in ("x", "y", "yaw_deg")):
            raise ValueError("点位包含无效坐标或朝向")
    return config, points


def inspect_plan(config, points):
    from precision_navigation import validate_settings
    validate_settings(config['precision'])
    if config['single_point_route'] not in (1, 2):
        raise ValueError('单点导航需要指定路线一或路线二')
    problems = list(config.get("pending", []))
    needed = {config["start"], config["branch"]["origin"], config["branch"]["toward"]}
    for section in ("common", "route1", "route2", "tail"):
        for index, step in enumerate(config[section]):
            keys = set(step) & {"move", "turn", "face", "face_point", "task"}
            if len(keys) != 1:
                raise ValueError("%s 第%d项需要一个明确动作" % (section, index + 1))
            if "move" in step:
                if step["mode"] not in MODES:
                    raise ValueError("不支持的移动方式")
                if step["move"] is None:
                    problems.append("%s 第%d项停车位置待指定" % (section, index + 1))
                else:
                    needed.add(step["move"])
            if "task" in step and step["task"] not in TASKS:
                raise ValueError("不支持的识别预留项：" + step["task"])
            if "face_point" in step:
                needed.add(step["face_point"])
                if not math.isfinite(float(step.get("offset", 0))):
                    raise ValueError("点位朝向角度需要为有限数值")
    problems.extend("缺少点位：" + name for name in sorted(needed - points.keys()))
    if config["task_dwell"] != 2.0:
        raise ValueError("识别预留停车时间需要为2秒")
    if config.get("recognition", {}).get("enabled"):
        from parking_ebike import validate_config
        parking = config["recognition"].get("parking_ebike")
        if parking is None:
            raise ValueError("停车区识别配置尚未填写")
        validate_config(parking)
        needed.update(parking.get("view_points", ["共享1", "共享2"]))
        problems.extend("缺少停车区点位：" + name for name in sorted(needed - points.keys()))
    return sorted(set(problems))


def route_steps(config, route):
    steps = config["common"] + config["route" + str(route)] + config["tail"]
    return [dict(step, turn=step["route2_turn"]) if route == 2 and "route2_turn" in step else dict(step) for step in steps]


def describe(config, points):
    for route in (1, 2):
        steps = route_steps(config, route)
        previous = points[config["start"]]
        distance = 0.0
        tasks = 0
        print("路线%d" % route)
        for index, step in enumerate(steps, 1):
            if "move" in step:
                target = points.get(step["move"])
                if target is not None and previous is not None:
                    distance += math.hypot(target["x"] - previous["x"], target["y"] - previous["y"])
                previous = target
            tasks += int("task" in step)
            print("%02d %s" % (index, json.dumps(step, ensure_ascii=False)))
        print("已知段总长 %.2f m，识别预留 %d 次，共 %.0f 秒" % (distance, tasks, tasks * 2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--start-args", action="store_true")
    args = parser.parse_args()
    config, points = load_plan(args.config)
    problems = inspect_plan(config, points)
    if args.start_args:
        if problems:
            raise SystemExit("\n".join(problems))
        from navigation_geometry import prepare_points
        start = prepare_points(config, points, args.config)[config["start"]]
        print("x:=%s y:=%s a:=%s" % (start["x"], start["y"], math.radians(start["yaw_deg"])))
        raise SystemExit(0)
    describe(config, points)
    for problem in problems:
        print("待确认：" + problem)
    raise SystemExit(1 if problems else 0)
