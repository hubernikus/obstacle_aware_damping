from enum import Enum, auto

###################
## MAGIC NUMBERS ##
###################

#controller
LAMBDA_MAX = 200.0 #max value of lambda, complicance in chosen direction
DIST_CRIT = 1.0 #dist when we stop considering obstacles

MAX_TAU_C = 500.0 #max possible torque


##################
## ENUM CLASSES ##
##################

class TypeOfDMatrix(Enum):
    DS_FOLLOWING = auto()
    OBS_PASSIVITY = auto()
    BOTH = auto()