# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The tests use tt-metal's pytest fixtures (mesh_device, device_params, ...): load tt-metal's conftest under
another name and take over its fixtures and hooks."""
import importlib.util
import os
import sys

from qwen36_resident import TT_METAL_HOME

sys.path.insert(0, TT_METAL_HOME)
_spec = importlib.util.spec_from_file_location("tt_metal_conftest", os.path.join(TT_METAL_HOME, "conftest.py"))
_tt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tt)
globals().update({k: v for k, v in vars(_tt).items() if not k.startswith("__")})
