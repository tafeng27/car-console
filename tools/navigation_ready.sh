source "$(dirname "$0")/session_common.sh"
/usr/bin/python3 "$WS/src/robot_navigation/scripts/navigation_trial.py" --check
check_idle
START_ARGS=$(/usr/bin/python3 "$WS/src/robot_navigation/scripts/navigation_plan.py" "$WS/src/robot_navigation/config/navigation_trial.yaml" --start-args)
read -r -a START_POSE <<< "$START_ARGS"
start_process bringup roslaunch real_robot_bringup field_bringup.launch localize:=true "$(camera_arg)" "${START_POSE[@]}"
wait_topic /amcl_pose
wait_topic /scan
start_process web /usr/bin/python3 "$WS/src/robot_navigation/scripts/web_control.py"
/usr/bin/python3 "$WS/tools/initialize_recorded_pose.py" --apply
echo "已启动定位与查看服务，等待导航指令。"
