#edit
NUM_G_AP = 3    #3
NUM_A_AP = 7    #7
NUM_AP = NUM_G_AP + NUM_A_AP
NUM_USERS = 10  #10

MAX_X = 500
MAX_Y = 500
MAX_H = 50
MIN_H = 1 #低于最低高度受到惩罚

MAX_V_X = 70
MAX_V_Y = 70
MAX_V_H = 20

LOS=-1.2
NLOS=-2.1
LOS_COEF1 = 11.95
LOS_COEF2 = 0.136

P_A=20
P_G=50
white_noise = 1e-6

punishment_coef = 0.1
cost_coef = 0.1
OVER_LOAD_PUNISHMENT = 2/punishment_coef
LOW_CHANNEL = 3/punishment_coef
CRASH = 100
LINK_COST = 1
min_capacity_uav = 0.1
min_capacity_user = 0.1



