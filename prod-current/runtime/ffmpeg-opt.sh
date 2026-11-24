#!/bin/bash
exec nice -n 5 /root/autodl-tmp/bin/ffmpeg-btbn/ffmpeg "$@"
