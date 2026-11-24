#!/bin/bash
WD=/root/autodl-tmp/bin/higgs-keepalive.sh
if setsid -f true >/dev/null 2>&1; then setsid -f bash "$WD" </dev/null >/dev/null 2>&1
else ( setsid bash "$WD" </dev/null >/dev/null 2>&1 & ); fi
