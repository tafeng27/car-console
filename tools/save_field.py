import argparse
import datetime
import json
import shutil
import subprocess
from pathlib import Path

import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src/robot_navigation/scripts'))
from teaching_capture import pair_manifest, saved_grid, validate_points, write_points
import rospy
from nav_msgs.msg import OccupancyGrid

workspace = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser()
parser.add_argument("--map-dir", type=Path, default=workspace / "src/robot_navigation/map")
map_dir = parser.parse_args().map_dir
map_dir.mkdir(parents=True, exist_ok=True)
points_file = map_dir / 'taught_points.yaml'
stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
nodes = subprocess.run(["rosnode", "list"], check=True, capture_output=True, text=True).stdout.splitlines()
mapping = "/slam_gmapping" in nodes or "/gmapping" in nodes
rospy.init_node('save_mapping_pair', anonymous=True)
grid = rospy.wait_for_message('/map', OccupancyGrid, timeout=10)
details_file = points_file.with_suffix('.json')
details = json.loads(details_file.read_text(encoding='utf-8'))
if mapping or "/amcl" in nodes:
    snapshot = map_dir / ("field_map_" + stamp)
    subprocess.run(["rosrun", "map_server", "map_saver", "-f", str(snapshot)], check=True, timeout=60)
    metadata = yaml.safe_load(snapshot.with_suffix(".yaml").read_text(encoding="utf-8"))
    metadata["image"] = snapshot.with_suffix(".pgm").name
    snapshot.with_suffix(".yaml").write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
    verified = validate_points(saved_grid(snapshot.with_suffix('.yaml')),
                               [{**point, 'recording_session': details['session']} for point in details['points']],
                               yaml.safe_load((workspace / 'src/robot_navigation/config/navigation_trial.yaml').read_text(encoding='utf-8'))['laser_geometry'])
    write_points(str(points_file), str(details_file), verified, stamp)
    shutil.copy2(snapshot.with_suffix(".pgm"), map_dir / "field_map.pgm")
    metadata["image"] = "field_map.pgm"
    (map_dir / "field_map.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
else:
    raise RuntimeError("保存需要正在运行的 gmapping 或 AMCL")

point_count = 0
if points_file.exists():
    points = yaml.safe_load(points_file.read_text(encoding="utf-8"))
    if not isinstance(points, dict):
        raise ValueError("点位文件需要 YAML 名称与坐标表")
    point_count = len(points)
    shutil.copy2(points_file, map_dir / ("taught_points_" + stamp + ".yaml"))
    details_file = points_file.with_suffix(".json")
    if details_file.exists():
        shutil.copy2(details_file, map_dir / ("taught_points_" + stamp + ".json"))
manifest = {**pair_manifest(str(map_dir), stamp), 'mode': 'mapping' if mapping else 'localize', 'points': point_count}
(map_dir / 'saved_pair.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
(map_dir / ("session_" + stamp + ".json")).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print("已保存地图：%s，点位数量：%d，地图与点位 SHA256 校验记录已生成。" % (snapshot.with_suffix(".yaml"), point_count))
