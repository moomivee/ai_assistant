"""Application package: disable unrelated telemetry before native imports."""

import os

os.environ["ORT_DISABLE_TELEMETRY"] = "1"
