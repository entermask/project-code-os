#!/bin/bash
# Keepalive watchdog: flock-singleton daemon that relaunches the higgs supervisord
# within ~20s if it dies (the failure a power blip causes when the container survives).
LOCK=/tmp/higgs-keepalive.lock
if command -v flock >/dev/null 2>&1; then exec 9>"$LOCK"; flock -n 9 || exit 0; fi
CONF=/root/autodl-tmp/supervisor/supervisord.conf
SUPD=$(command -v supervisord || echo /usr/bin/supervisord)
LOG=/root/autodl-tmp/logs/keepalive.log
while true; do
  if ! pgrep -f "supervisord -c $CONF" >/dev/null 2>&1; then
    echo "$(date '+%F %T') supervisord down -> relaunch" >> "$LOG"
    rm -f /root/autodl-tmp/supervisor/supervisor.sock /root/autodl-tmp/supervisor/supervisord.pid 2>/dev/null
    # Port 8000 con TIME_WAIT tu tien trinh vua chet -> sgl-omni IM LANG nhay sang
    # port ngau nhien, bridge khong bao gio thay engine (box bao RUNNING nhung vo
    # dung). Cho bind duoc kieu vanilla roi moi start. Da gap that tren box 5.
    for _ in $(seq 1 60); do
      /root/miniconda3/bin/python3 -c \
        'import socket; s=socket.socket(); s.bind(("127.0.0.1",8000)); s.close()' 2>/dev/null && break
      sleep 2
    done
    "$SUPD" -c "$CONF" >> "$LOG" 2>&1
  fi
  sleep 20
done
