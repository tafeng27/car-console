# 保存地图与点位；--keep 保持程序运行。
source "$(dirname "$0")/session_common.sh"
/usr/bin/python3 "$WS/tools/save_field.py"
if [ "${1:-}" != '--keep' ]; then
    stop_session
fi
