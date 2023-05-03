import numpy as np
from collections import defaultdict

samp_rate = 200

training_config = {'XGB_config': {'num_trees':100, 'trees_depth':15}}

region_map = defaultdict(lambda: np.nan)
region_map["Right Lower Jaw Top"] = 0.0
region_map["Right Lower Jaw Front"] = 1.0
region_map["Right Lower Jaw Back"] = 2.0
region_map["Lower Incisors Front"] = 3.0
region_map["Lower Incisors Back"] = 4.0
region_map["Left Lower Jaw Top"] = 5.0
region_map["Left Lower Jaw Front"] = 6.0
region_map["Left Lower Jaw Back"] = 7.0
region_map["Right Upper Jaw Top"] = 8.0
region_map["Right Upper Jaw Front"] = 9.0
region_map["Right Upper Jaw Back"] = 10.0
region_map["Upper Incisors Front"] = 11.0
region_map["Upper Incisors Back"] = 12.0
region_map["Left Upper Jaw Top"] = 13.0
region_map["Left Upper Jaw Front"] = 14.0
region_map["Left Upper Jaw Back"] = 15.0


region_map_merge = defaultdict(lambda: np.nan)
region_map_merge["Right Lower Jaw Top"] = 0.0
region_map_merge["Right Lower Jaw Back"] = 0.0
region_map_merge["Right Lower Jaw Front"] = 1.0
region_map_merge["Right Upper Jaw Front"] = 1.0
region_map_merge["Lower Incisors Front"] = 2.0
region_map_merge["Upper Incisors Front"] = 2.0
region_map_merge["Lower Incisors Back"] = 3.0
region_map_merge["Left Lower Jaw Top"] = 4.0
region_map_merge["Left Lower Jaw Back"] = 4.0
region_map_merge["Left Lower Jaw Front"] = 5.0
region_map_merge["Left Upper Jaw Front"] = 5.0
region_map_merge["Right Upper Jaw Top"] = 6.0
region_map_merge["Right Upper Jaw Back"] = 6.0
region_map_merge["Upper Incisors Back"] = 7.0
region_map_merge["Left Upper Jaw Top"] = 8.0
region_map_merge["Left Upper Jaw Back"] = 8.0

# region_map = {
#         "Left Lower Jaw Front":1,
#         "Left Lower Jaw Top":2,
#         "Left Lower Jaw Back":3,
#         "Left Upper Jaw Front":4,
#         "Left Upper Jaw Top":5,
#         "Left Upper Jaw Back":6,
#         "Right Lower Jaw Front":7,
#         "Right Lower Jaw Top":8,
#         "Right Lower Jaw Back":9,
#         "Right Upper Jaw Front":10,
#         "Right Upper Jaw Top":11,
#         "Right Upper Jaw Back":12,
#         "Lower Incisors Front":13,
#         "Lower Incisors Back":14,
#         "Upper Incisors Front":15,
#         "Upper Incisors Back":16
#     }


unlabeled_folders = [
    'sample', 
    'S4-S2-F-R-AW-26-M-1-AG',
    'S5-S2-M-R-AW-40-M-2-AG',
    'S7-S1-M-R-AW-31-M-2-AG',
    'S7-S2-M-R-AW-31-M-2-AG',
    'S7-S3-M-R-AW-31-M-2-AG',
    'S8-S1-M-R-AW-31-M-2-AG',
    'S8-S2-M-R-AW-31-M-2-AG',
    'S9-S2-M-R-AW-30-M-2-AG',
    'S10-S1-M-R-AW-30-E-3-AG',
    'S11-S1-F-R-AW-30-M-3-AG',
    'S13-S1-F-R-AW-30-M-4-AG',
    'S13-S3-F-R-AW-30-M-4-AG',
    'S14-S2-F-R-AW-35-M-5-AG',
    'S15-S2-F-R-AW-30-M-5-AG',
    'S16-S1-F-R-AW-28-M-5-AG',
    'S16-S2-F-R-AW-28-M-5-AG',
    'S16-S3-F-R-AW-28-M-5-AG',
    'S17-S2-M-R-AW-40-M-2-AG',
    ]


patient_out_not_include = ['P8', 'P16']

session_out_not_include = ['P4', 'P7', 'P8', 'P9', 'P14', 'P15', 'P16', 'P17']
