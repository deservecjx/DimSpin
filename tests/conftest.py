"""Shared test configuration."""

import os

# Keep plotting tests headless on every platform.
os.environ.setdefault("MPLBACKEND", "Agg")
