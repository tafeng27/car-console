import argparse
import json
import math
from pathlib import Path
import sys

import rospy
import tf
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan


workspace = Path.home() / 'final_ws'
sys.path.insert(0, str(workspace / 'src/robot_navigation/scripts'))
from navigation_geometry import prepare_points
from navigation_plan import load_plan
from precision_navigation import ReferenceMatcher, scan_sample


parser = argparse.ArgumentParser()
parser.add_argument('--apply', action='store_true')
parser.add_argument('--point')
parser.add_argument('--heading-free', action='store_true')
parser.add_argument('--local-reference', action='store_true')
arguments = parser.parse_args()
config_path = workspace / 'src/robot_navigation/config/navigation_trial.yaml'
config, recorded = load_plan(config_path)
goal = prepare_points(config, recorded, config_path)[arguments.point or config['start']]
rospy.init_node('initialize_recorded_pose', anonymous=True)
listener = tf.TransformListener()
listener.waitForTransform('map', 'base_footprint', rospy.Time(0), rospy.Duration(5))
rospy.sleep(1)
scan = rospy.wait_for_message('/scan', LaserScan, timeout=3)
points, mapped = scan_sample(listener, scan)
pose, quality = ReferenceMatcher(goal).match(points, mapped)
distance_limit = 0.15 if arguments.local_reference else 0.03
if math.dist(pose[:2], (goal['x'], goal['y'])) > distance_limit or (not arguments.heading_free and abs(math.atan2(math.sin(pose[2] - goal['yaw_rad']), math.cos(pose[2] - goal['yaw_rad']))) > math.radians(5)):
    raise RuntimeError('记录激光检查表明当前车辆不在起停点或朝向不符')
if arguments.local_reference:
    import numpy as np
    from precision_navigation import MapMatcher
    from teaching_capture import saved_grid

    if quality['inlier_fraction'] < 0.90:
        raise RuntimeError('局部定位复核需要至少90%的记录激光有效比例')
    grid = saved_grid(workspace / 'src/robot_navigation/map/field_map.yaml')
    q = grid.info.origin.orientation
    origin_yaw = tf.transformations.euler_from_quaternion((q.x, q.y, q.z, q.w))[2]
    matcher = MapMatcher(np.asarray(grid.data).reshape(grid.info.height, grid.info.width), grid.info.resolution,
                          (grid.info.origin.position.x, grid.info.origin.position.y, origin_yaw))
    independent = matcher.project({'name': goal['name'], 'x': pose[0], 'y': pose[1],
                                   'yaw_deg': math.degrees(pose[2]), 'scan_base': points})
    if math.dist(pose[:2], (independent['x'], independent['y'])) > 0.03:
        raise RuntimeError('地图与记录激光未能共同确认当前位置')
if arguments.apply:
    publisher = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=1, latch=True)
    deadline = rospy.Time.now() + rospy.Duration(3)
    while publisher.get_num_connections() == 0 and rospy.Time.now() < deadline:
        rospy.sleep(0.05)
    if publisher.get_num_connections() == 0:
        raise RuntimeError('AMCL未连接初始位置话题')
    message = PoseWithCovarianceStamped()
    message.header.frame_id = 'map'
    message.header.stamp = rospy.Time.now()
    message.pose.pose.position.x, message.pose.pose.position.y = pose[:2]
    message.pose.pose.orientation.z, message.pose.pose.orientation.w = math.sin(pose[2] / 2), math.cos(pose[2] / 2)
    message.pose.covariance[0] = message.pose.covariance[7] = 0.02 ** 2
    message.pose.covariance[35] = math.radians(2) ** 2
    publisher.publish(message)
    rospy.sleep(0.5)
print(json.dumps({'pose': pose, 'mapped_before': mapped, 'quality': quality, 'applied': arguments.apply}, ensure_ascii=False))
