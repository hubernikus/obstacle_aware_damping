from enum import Enum, auto
import numpy as np

###################
## MAGIC NUMBERS ##
###################

#controller
LAMBDA_MAX = 200.0 #max value of lambda, complicance in chosen direction

DIST_CRIT = 0.8 #dist when we stop considering obstacles
D_SCAL = 1 #from lukas theory

EPS_CONVERGENCE = 0.02 #zone around atractor to improve stability

MAX_TAU_C = 500.0 #max possible torque #500 works well, 100 also, but less damped against obs

NOISE_MAGN_POS = 0.0 #0.5 still good
NOISE_MAGN_VEL = 5. #1 still good

#tank system
S_MAX = 100.0
DELTA_S = 0.1*S_MAX #smoothness parameter
DELTA_Z = 0.01      #smoothness parameter

#display
QOLO_LENGHT_X = 0.4

#EVERYTHING
EPSILON = 1e-6

##################
## ENUM CLASSES ##
##################

class TypeOfDMatrix(Enum):
    DS_FOLLOWING = auto()
    OBS_PASSIVITY = auto()
    BOTH = auto()