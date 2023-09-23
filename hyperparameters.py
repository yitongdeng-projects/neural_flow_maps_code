# 
# simulation hyperparameters
res_x = 128
res_y = 128
res_z = 256
visualize_dt = 0.1
reinit_every = 20
ckpt_every = 10
CFL = 0.5
from_frame = 0
total_frames = 500
BFECC_clamp = True
exp_name = "four_vortices"

# learning hyperparameters
min_res = (16, 16, 32) # encoder base resolutions
num_levels = 4 # number of refining levels
feat_dim = 4 # feature vector size (per anchor vector)
activate_threshold = 0.045 # smaller means more cells are activated
N_iters = 200
N_batch = 240000
success_threshold = 2e-6 # smaller means later termination
num_chunks = 2 # query buffer in chunks (as small as machine memory permits)

