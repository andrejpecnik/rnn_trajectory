"""Global constants shared by the simulator, the dataset, the models and the game.

All distances are in "world units" (1 unit = 1 pixel in the game window),
all times are in seconds and all angles are in radians.
"""

import math

# --- Arena -------------------------------------------------------------------
ARENA_W = 1000.0
ARENA_H = 700.0

# Static obstacles: (x, y, width, height), top-left corner in world coordinates.
OBSTACLES = (
    (210.0, 140.0, 130.0, 36.0),
    (660.0, 140.0, 130.0, 36.0),
    (210.0, 524.0, 130.0, 36.0),
    (660.0, 524.0, 130.0, 36.0),
    (470.0, 290.0, 60.0, 120.0),
    (90.0, 315.0, 36.0, 70.0),
    (874.0, 315.0, 36.0, 70.0),
)

# --- Time ----------------------------------------------------------------------
SIM_HZ = 60
SIM_DT = 1.0 / SIM_HZ           # physics step
MODEL_STRIDE = 2                # the predictor sees every 2nd physics tick
MODEL_DT = SIM_DT * MODEL_STRIDE  # = 1/30 s

# --- Prediction window -----------------------------------------------------------
HISTORY = 30                    # 30 model steps = 1.0 s of observed past
HORIZON = 45                    # 45 model steps = 1.5 s predicted future

# --- Tank physics ------------------------------------------------------------------
TANK_RADIUS = 18.0
TANK_MAX_SPEED = 160.0          # units / s forward
TANK_MAX_REVERSE = 100.0        # units / s backward
TANK_ACCEL = 260.0              # units / s^2 when speeding up
TANK_BRAKE = 420.0              # units / s^2 when slowing down / reversing
TANK_TURN_RATE = 2.4            # rad / s for hull rotation at full turn input
TURRET_TURN_RATE = 4.5          # rad / s for turret rotation
TANK_HP = 5

# --- Weapons -------------------------------------------------------------------------
BULLET_SPEED = 420.0            # units / s
BULLET_RADIUS = 4.0
BULLET_LIFETIME = 3.0           # s
MUZZLE_LENGTH = 28.0            # distance from tank centre to the barrel tip
RELOAD_TIME = 0.75              # s between shots

# --- Ray sensors used as model features ------------------------------------------------
RAY_MAX = 300.0                 # rays are clipped at this length
RAY_ANGLES = (0.0, math.pi / 4, -math.pi / 4)   # front, front-left, front-right

# --- Feature normalisation (fixed constants, no dataset statistics needed) -------------
STEP_SCALE = TANK_MAX_SPEED * MODEL_DT  # max displacement per model step (~5.33)
LOSS_SCALE = 50.0                       # loss is measured in units of 50 world units
N_FEATURES = 11
