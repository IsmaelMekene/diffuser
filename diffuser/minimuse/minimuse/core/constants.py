# simulation
SIM_STEPS = 10
RENDER_STEPS = 4
# WARMUP_STEPS = 100
WARMUP_STEPS = 1000

# controller rate - second
CONTROLLER_DT = 0.1
# (linvel, angvel) - meter / second
MAX_TOOL_VELOCITY = (0.025, 0.25)

# camera
REALSENSE_RESOLUTION = (1280, 720)
# same 16/9 aspect ratio than original resolution
RENDER_RESOLUTION = (448, 252)
# 4/3 crop of the image
REALSENSE_CROP = (240, 180)
REALSENSE_CROP_Y = 0
REALSENSE_FOV = 42.5
