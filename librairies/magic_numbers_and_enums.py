from enum import Enum, auto

###################
## MAGIC NUMBERS ##
###################

#controller
LAMBDA_MAX = 200.0 #max value of lambda, complicance in chosen direction
DIST_CRIT = 0.8 #dist when we stop considering obstacles

EPS_CONVERGENCE = 0.02 #zone around atractor to improve stability

MAX_TAU_C = 500.0 #max possible torque

#EVERYTHING
DIM = 2
EPSILON = 1e-6


##################
## ENUM CLASSES ##
##################

class TypeOfDMatrix(Enum):
    DS_FOLLOWING = auto()
    OBS_PASSIVITY = auto()
    BOTH = auto()