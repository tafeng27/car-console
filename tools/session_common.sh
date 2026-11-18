set -e
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/noetic/setup.bash
source "$WS/devel/setup.bash"
set -o pipefail
DATA_DIR="$WS/data/record"
LOG_DIR="$DATA_DIR/logs"
SESSION_DIR="$WS/.work/session"
mkdir -p "$DATA_DIR" "$LOG_DIR" "$SESSION_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
export PYTHONUNBUFFERED=1
export YOLO_AUTOINSTALL=false
export YOLO_CONFIG_DIR="$WS/.work/yolo"
export TMPDIR="$WS/.work"
export CAR_RECORD_DIR="$DATA_DIR"
mkdir -p "$YOLO_CONFIG_DIR/Ultralytics"

stop_session() {
    local pidfile pid
    if compgen -G "$SESSION_DIR/*.pid" >/dev/null; then
        timeout 5 rostopic pub -1 /cmd_vel geometry_msgs/Twist '{}' >/dev/null
        for pidfile in "$SESSION_DIR"/*.pid; do
            pid=$(< "$pidfile")
            if kill -0 "$pid" 2>/dev/null; then
                kill -INT -- "-$pid"
            fi
        done
        for attempt in $(seq 1 15); do
            local active=0
            for pidfile in "$SESSION_DIR"/*.pid; do
                pid=$(< "$pidfile")
                if kill -0 "$pid" 2>/dev/null && [ "$(ps -o stat= -p "$pid" | cut -c1)" != Z ]; then
                    active=1
                fi
            done
            [ "$active" -eq 0 ] && break
            sleep 1
        done
        for pidfile in "$SESSION_DIR"/*.pid; do
            pid=$(< "$pidfile")
            if kill -0 "$pid" 2>/dev/null && [ "$(ps -o stat= -p "$pid" | cut -c1)" != Z ]; then
                echo "程序尚未退出，请检查 PID $pid"
                return 1
            fi
        done
        find "$SESSION_DIR" -maxdepth 1 -name '*.pid' -delete
    fi
}

check_idle() {
    if pgrep -af '(^|/)(roslaunch|roscore)( |$)'; then
        echo "已有 ROS 启动程序，请关闭该程序后继续。"
        return 1
    fi
}

start_process() {
    local name=$1
    shift
    setsid "$@" > "$LOG_DIR/${name}_$STAMP.log" 2>&1 < /dev/null &
    echo "$!" > "$SESSION_DIR/$name.pid"
}

camera_arg() {
    if lsusb | grep -qi '2bc5'; then
        echo 'use_camera:=true'
    else
        echo 'use_camera:=false'
    fi
}

wait_topic() {
    for attempt in $(seq 1 12); do
        if timeout 3 rostopic echo -n 1 "$1" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    echo "没有收到话题数据：$1"
    return 1
}
