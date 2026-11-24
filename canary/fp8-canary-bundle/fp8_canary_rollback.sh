#!/bin/bash
# fp8_canary_rollback.sh — manual rollback of the FP8 canary on PROD box 22405: remove the env block, point
# sglang-omni-k16 back at sglang-omni-k16-e161d2f, restart ONLY higgs_sglang, wait until ready. Drain first if you can.
. /root/autodl-tmp/fp8-canary-bundle/fp8_canary_lib.sh
pkill -f "^bash /root/autodl-tmp/fp8-canary-bundle/fp8_canary_switch.sh" 2>/dev/null
rollback "manual rollback"
cat $STATE
