#!/usr/bin/env bash
# Khoi dong service Higgs (supervisord cua minh) sau khi container boot.
# Idempotent: goi bao nhieu lan cung an toan. Duoc /etc/autodl.sh goi luc boot,
# nhung chay tay cung duoc khi can cuu mot box vua chet.
CONF=/root/autodl-tmp/supervisor/supervisord.conf
LOG=/root/autodl-tmp/logs/boot.log
SOCK=/root/autodl-tmp/supervisor/supervisor.sock
PIDFILE=/root/autodl-tmp/supervisor/supervisord.pid

mkdir -p /root/autodl-tmp/logs
exec >>"$LOG" 2>&1
echo "== $(date '+%F %T') boot hook chay"

# Duong dan khac nhau giua cac box (/usr/bin vs /usr/local/bin) -> TU DO, khong
# hardcode. Thieu binary thi thoat ngay, tuyet doi khong dong vao socket/pid.
SUPERVISORD=$(command -v supervisord || echo /usr/local/bin/supervisord)
SUPERVISORCTL=$(command -v supervisorctl || echo /usr/local/bin/supervisorctl)
if [ ! -x "$SUPERVISORD" ]; then
    echo "LOI: khong tim thay supervisord -> thoat, khong dong gi"
    exit 1
fi

# CHAN 1: co tien trinh supervisord cua minh dang song -> khong lam gi. Kiem tra
# bang process chu KHONG bang supervisorctl, vi supervisorctl loi (sai path/socket)
# se bi hieu nham la "chua chay" roi xoa nham socket cua tien trinh dang song.
if ps -eo args | grep -q "[s]upervisord -c $CONF"; then
    echo "supervisord da chay (theo process) -> bo qua"
    exit 0
fi
# CHAN 2: process khong thay nhung ctl van noi chuyen duoc -> van coi la dang chay.
if [ -x "$SUPERVISORCTL" ] && "$SUPERVISORCTL" -c "$CONF" status >/dev/null 2>&1; then
    echo "supervisord da chay (theo supervisorctl) -> bo qua"
    exit 0
fi

# Cho GPU san sang (toi da ~150s) truoc khi start engine.
for _ in $(seq 1 30); do
    nvidia-smi -L >/dev/null 2>&1 && break
    sleep 5
done
nvidia-smi -L >/dev/null 2>&1 || echo "CANH BAO: GPU chua san sang, van thu start"

# Toi day chac chan khong co supervisord cua minh -> socket/pid la rac tu lan chet
# truoc, xoa di neu khong supervisord tu choi khoi dong.
rm -f "$SOCK" "$PIDFILE"

# Port 8000 con TIME_WAIT -> sgl-omni IM LANG nhay sang port ngau nhien, bridge
# khong bao gio thay engine. Cho toi khi bind duoc bang dung kieu bind vanilla.
for _ in $(seq 1 60); do
    /root/miniconda3/bin/python3 -c \
        'import socket; s=socket.socket(); s.bind(("127.0.0.1",8000)); s.close()' \
        2>/dev/null && break
    sleep 2
done

if "$SUPERVISORD" -c "$CONF"; then
    echo "supervisord started ($SUPERVISORD)"
else
    echo "LOI: khong start duoc supervisord"
    exit 1
fi
