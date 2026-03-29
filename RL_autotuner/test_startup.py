#!/usr/bin/env python3
import sys
print("1. Script started", flush=True)

import os
print("2. os imported", flush=True)

sys.path.insert(0, "/home/dqy/aeroplanax/new/20251215最新代码库/RL_autotuner")
print("3. Path added", flush=True)

from config_patcher import load_config
print("4. config_patcher imported", flush=True)

print("5. All imports successful!", flush=True)
