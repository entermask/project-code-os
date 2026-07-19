#!/bin/bash
LOCK=/tmp/higgs-keepalive.lock
if command -v flock >/dev/null 2>&1; then exec 9>"$LOCK"; flock -n 9 || exit 0; fi
CONF=/root/autodl-tmp/supervisor/supervisord.conf
SUPD=$(command -v supervisord || echo /usr/bin/supervisord)
LOG=/root/autodl-tmp/logs/keepalive.log
while true; do
  if ! pgrep -f "supervisord -c $CONF" >/dev/null 2>&1; then
    echo "$(date '+%F %T') supervisord down -> relaunch" >> "$LOG"
    rm -f /root/autodl-tmp/supervisor/supervisor.sock /root/autodl-tmp/supervisor/supervisord.pid 2>/dev/null
    "$SUPD" -c "$CONF" >> "$LOG" 2>&1
  fi
  sleep 20
done
