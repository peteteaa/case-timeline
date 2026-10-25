#!/bin/bash
# Let NemoClaw sandboxes reach the loopback-only vLLM server.
# Sandboxes resolve host.openshell.internal to Docker's host-gateway (docker0, 172.17.0.1),
# so listen only there and forward to 127.0.0.1:8000. Nothing listens on LAN/Tailscale.
exec socat TCP-LISTEN:8000,bind=172.17.0.1,reuseaddr,fork TCP:127.0.0.1:8000
