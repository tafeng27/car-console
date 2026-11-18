# 启动建图与网页。
source "$(dirname "$0")/session_common.sh"
check_idle
/usr/bin/python3 "$WS/tools/begin_mapping.py"
start_process bringup roslaunch real_robot_bringup field_bringup.launch localize:=false "$(camera_arg)" "$@"
wait_topic /scan
wait_topic /odom
start_process slam roslaunch robot_navigation slam_tuned.launch
wait_topic /map
start_process web /usr/bin/python3 "$WS/src/robot_navigation/scripts/web_control.py"
sleep 2
curl --fail --silent http://127.0.0.1:8090/api/state
/usr/bin/python3 "$WS/src/robot_navigation/scripts/mapping_check.py"
echo "保存并结束建图：bash $WS/tools/map_save.sh"
