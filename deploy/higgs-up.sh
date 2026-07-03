#!/bin/bash
# Idempotent bring-up: daemonize the keepalive watchdog (which in turn ensures supervisord
# -> higgs_sglang + higgs_api). Used by /etc/autodl.sh at boot and for manual use.
# Does NOT free the GPU; for a stuck/OOM/crash-loop use higgs-restart.sh instead.
WD=/root/autodl-tmp/bin/higgs-keepalive.sh
if setsid -f true >/dev/null 2>&1; then setsid -f bash "$WD" </dev/null >/dev/null 2>&1
else ( setsid bash "$WD" </dev/null >/dev/null 2>&1 & ); fi
