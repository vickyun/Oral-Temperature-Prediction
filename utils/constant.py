from datetime import datetime

OUTPUT = "out/" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '/'

# Tracking settings
n = "1"
RUN_NAME = f"Train on {n} feature(s)"

FRAC = [0.8, 0.0, 0.2]
NB_ITER = 1000
ALPHA = 0.005
SEED = 29

# List features used for training
FEATURES = ['canthiMax1']

# ['Gender', 'Age', 'Ethnicity', 'T_atm', 'Humidity', 'Distance',
#        'T_offset1', 'Max1R13_1', 'Max1L13_1', 'aveAllR13_1', 'aveAllL13_1',
#        'T_RC1', 'T_RC_Dry1', 'T_RC_Wet1', 'T_RC_Max1', 'T_LC1', 'T_LC_Dry1',
#        'T_LC_Wet1', 'T_LC_Max1', 'RCC1', 'LCC1', 'canthiMax1', 'canthi4Max1',
#        'T_FHCC1', 'T_FHRC1', 'T_FHLC1', 'T_FHBC1', 'T_FHTC1', 'T_FH_Max1',
#        'T_FHC_Max1', 'T_Max1', 'T_OR1', 'T_OR_Max1']

