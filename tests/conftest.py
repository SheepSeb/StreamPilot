import os

# Headless offscreen rendering for the pixel and detection tests.
os.environ.setdefault("MUJOCO_GL", "egl")
