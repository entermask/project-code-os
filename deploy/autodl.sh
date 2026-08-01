#!/bin/bash
# Hook khoi dong cua AutoDL/GPUhub: /init/bin/customer.cmd.sh chay file nay moi
# lan container boot. Logic that de o data disk (/root/autodl-tmp) nen khong mat
# khi system disk bi dung lai; file nay chi la mot dong moi.
# Chay nen de khong chan qua trinh boot.
if [ -x /root/autodl-tmp/bin/higgs-boot.sh ]; then
    setsid nohup /root/autodl-tmp/bin/higgs-boot.sh >/dev/null 2>&1 &
fi
exit 0
