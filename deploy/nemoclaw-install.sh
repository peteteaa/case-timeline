#!/bin/bash
# Rerun NemoClaw install without Express mode, so it can use the existing vLLM at 172.17.0.1:8000.
# NEMOCLAW_FRESH=1 discards the failed Express session (it carries vllmInstallModel, which would
# make a resume replay the managed-vLLM install).
exec sg docker -c "NEMOCLAW_NO_EXPRESS=1 NEMOCLAW_FRESH=1 bash /home/dell/NemoClaw/nemoclaw-hosted-installer.sh"
