from enum import Enum, auto

###################
## MAGIC NUMBERS ##
###################

#controller
LAMBDA_MAX = 200.0 #max value of lambda, complicance in chosen direction
DIST_CRIT = 0.5 #dist when we stop considering obstacles

MAX_TAU_C = 500.0 #max possible torque

#EVERYTHING
DIM = 2


##################
## ENUM CLASSES ##
##################

class TypeOfDMatrix(Enum):
    DS_FOLLOWING = auto()
    OBS_PASSIVITY = auto()
    BOTH = auto()