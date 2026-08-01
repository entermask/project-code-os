#!/bin/bash
# Boot hook cua GPUhub/AutoDL: /init/bin/customer.cmd.sh chay file nay moi lan
# container khoi dong. Truoc day file KHONG ton tai (log boot ghi "No such file")
# nen sau moi lan instance restart service nam chet im — box 5 da dinh dung vay.
# Logic that o data disk vi system disk co the bi dung lai.
if [ -x /root/autodl-tmp/bin/higgs-up.sh ]; then
    setsid nohup bash /root/autodl-tmp/bin/higgs-up.sh >/dev/null 2>&1 &
fi
exit 0
