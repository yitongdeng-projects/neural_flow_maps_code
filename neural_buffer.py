# 
from hyperparameters import *
from models import *
from optimizers import *
import time
from io_utils import *
import torch

class NeuralBuffer3():

    def __init__(self, res_x, res_y, res_z, dx):
        self.use_importance_sample = True # if using importance sample
        self.res_x = res_x
        self.res_y = res_y
        self.res_z = res_z
        self.dx = dx
        self.model = SNF3T()
        self.model.fp.reinit(None)
        self.model_prev = SNF3T(prev_model = True) # a model that stores the previous instance
        # no need to init model_prev
        self.loss = lambda x, y : torch.mean((x - y) ** 2)
        self.lr = 0.01
        self.N_iters = N_iters
        self.N_batch = N_batch
        # for early-termination
        self.success_threshold = success_threshold
        self.N_success = 4
        # for keeping track of the conversion of actual time and normalized time
        self.curr_t_ratio = None
        self.prev_t_ratio = None
        self.sizing = None
        self.magnitude_scale = 1.
        
    def set_magnitude_scale(self, max_speed):
        self.magnitude_scale = 1./max_speed
        
    def paint_active(self, display):
        self.model.fp.paint_active(display)

    def reinit(self, sizing):
        self.model.fp.reinit(sizing)
        self.sizing = sizing.to_torch()
    
    def grow(self, sizing):
        self.model.fp.grow(sizing)
        self.sizing = torch.maximum(self.sizing, sizing.to_torch()) 

    def compute_importance(self):
        if self.use_importance_sample:
            min_val = (self.model.fp.res[0][1]-1) * self.model.fp.activate_threshold
            max_val = (self.model.fp.res[-1][1]-1) * self.model.fp.activate_threshold
            return torch.clamp(self.sizing, min_val, max_val)
        else:
            return None

    # compute the midpoint timestamps given dts
    def comp_mid_ts(self, dts):
        mid_ts = torch.zeros_like(dts)
        for i in range(1, mid_ts.shape[0]):
            mid_ts[i] = mid_ts[i-1] + dts[i-1]
        mid_ts += 0.5 * dts
        return mid_ts

    def gen_xyz_coords(self, N, importance=None):
        num_x, num_y, num_z = self.res_x, self.res_y, self.res_z
        num_yz = num_y * num_z
        num_xyz = num_x * num_y * num_z
        if importance is None:
            indices = torch.randint(0, num_xyz, size = (N,)) # with replacement
        else:
            indices = torch.multinomial(importance.flatten(), N, replacement=False) # with replacement
        indices_x = indices // num_yz
        indices_yz = indices % num_yz
        indices_y = indices_yz // num_z
        indices_z = indices_yz % num_z
        return torch.stack([indices_x, indices_y, indices_z], dim = -1)

    def sample(self, curr_u_x, curr_u_y, curr_u_z, 
            init_u_x, init_u_y, init_u_z, 
            mid_ts, importance = None):

        offsets = torch.randint(0, 2, (3,)) # 3 rand numbers in [0, 1], decide whether left- or right-align
        n_frames = mid_ts.shape[0]

        curr_N_batch = int(self.N_batch/(n_frames))
        prev_N_batch = self.N_batch - curr_N_batch

        curr_coord_t = self.curr_t_ratio * mid_ts[-1] * torch.ones((curr_N_batch,1))
        init_coord_t = torch.zeros((curr_N_batch,1))
    
        x_sampled = self.gen_xyz_coords(curr_N_batch, importance)

        y_sampled = x_sampled.detach().clone()
        z_sampled = x_sampled.detach().clone()

        x_sampled[:, 0] += offsets[0] # add 1 to x (right-align), or not (left-align)
        y_sampled[:, 1] += offsets[1]
        z_sampled[:, 2] += offsets[2] 
        
        curr_u_x_sampled = curr_u_x[x_sampled[:, 0], x_sampled[:, 1], x_sampled[:, 2]][:, None, None]
        coord_x_sampled = self.dx * torch.cat((x_sampled[:, [0]], x_sampled[:, [1]] + 0.5, x_sampled[:, [2]] + 0.5), dim = -1)
        curr_coord_xt_sampled = torch.cat((coord_x_sampled, curr_coord_t), dim = -1)[:, None, :]

        curr_u_y_sampled = curr_u_y[y_sampled[:, 0], y_sampled[:, 1], y_sampled[:, 2]][:, None, None]
        coord_y_sampled = self.dx * torch.cat((y_sampled[:, [0]] + 0.5, y_sampled[:, [1]], y_sampled[:, [2]] + 0.5), dim = -1)
        curr_coord_yt_sampled = torch.cat((coord_y_sampled, curr_coord_t), dim = -1)[:, None, :]

        curr_u_z_sampled = curr_u_z[z_sampled[:, 0], z_sampled[:, 1], z_sampled[:, 2]][:, None, None]
        coord_z_sampled = self.dx * torch.cat((z_sampled[:, [0]] + 0.5, z_sampled[:, [1]] + 0.5, z_sampled[:, [2]]), dim = -1)
        curr_coord_zt_sampled = torch.cat((coord_z_sampled, curr_coord_t), dim = -1)[:, None, :]

        curr_coord_sampled = torch.cat((curr_coord_xt_sampled, curr_coord_yt_sampled, curr_coord_zt_sampled), dim = 1) # [N, 3, 4]
        curr_u_sampled = torch.cat((curr_u_x_sampled, curr_u_y_sampled, curr_u_z_sampled), dim = 1) # [N, 3, 1]

        coord_sampled = curr_coord_sampled # [N, 3, 4]
        u_sampled = curr_u_sampled # [N, 3, 1]        

        # done sampling from data, now begin to sample from prev model
        if n_frames > 1:
            x_sampled = self.gen_xyz_coords(prev_N_batch, importance)
            t_sampled = torch.randint(0, torch.numel(mid_ts[:-1]), size = (prev_N_batch,1))
            xt_sampled = torch.cat([x_sampled, t_sampled], dim = -1)

            yt_sampled = xt_sampled.detach().clone()
            zt_sampled = xt_sampled.detach().clone()

            xt_sampled[:, 0] += offsets[0]
            yt_sampled[:, 1] += offsets[1]
            zt_sampled[:, 2] += offsets[2]

            t_sampled = mid_ts[xt_sampled[:, [-1]]][:, None, :].expand(-1, 3, -1) # [N, 3, 1]
            coord_x_sampled = self.dx * torch.cat((xt_sampled[:, [0]],
                                                (xt_sampled[:, [1]] + 0.5),
                                                xt_sampled[:, [2]] + 0.5), dim = -1)[:, None, :] # [N, 1, 3]
            coord_y_sampled = self.dx * torch.cat(((yt_sampled[:, [0]] + 0.5),
                                                yt_sampled[:, [1]],
                                                yt_sampled[:, [2]] + 0.5), dim = -1)[:, None, :] # [N, 1, 3]
            coord_z_sampled = self.dx * torch.cat(((zt_sampled[:, [0]] + 0.5),
                                                zt_sampled[:, [1]] + 0.5,
                                                zt_sampled[:, [2]]), dim = -1)[:, None, :] # [N, 1, 3]

            coord_xyz_sampled = torch.cat((coord_x_sampled, coord_y_sampled, coord_z_sampled), dim = 1) # [N, 3, 3]
            # evaluate using previous time scale
            prev_u_sampled = self.model_prev(torch.cat((coord_xyz_sampled,\
                                            self.prev_t_ratio * t_sampled), dim = -1)) # [N, 3, 4]
            # but associate the result with current time scale
            prev_coord_sampled = torch.cat((coord_xyz_sampled, self.curr_t_ratio * t_sampled), dim = -1) # [N, 3, 4]

            coord_sampled = torch.cat((coord_sampled, prev_coord_sampled), dim = 0)
            u_sampled = torch.cat((u_sampled, prev_u_sampled), dim = 0)

        return coord_sampled, u_sampled

    # train SNF to store velocity
    def store_u(self, curr_u_x, curr_u_y, curr_u_z,\
                    init_u_x, init_u_y, init_u_z, dts):
        # rescale
        curr_u_x *= self.magnitude_scale
        curr_u_y *= self.magnitude_scale
        curr_u_z *= self.magnitude_scale
        init_u_x *= self.magnitude_scale
        init_u_y *= self.magnitude_scale
        init_u_z *= self.magnitude_scale
        
        self.curr_t_ratio = 1./dts.sum()
        self.mid_ts = self.comp_mid_ts(dts)

        if (self.prev_t_ratio is not None) and self.prev_t_ratio / self.curr_t_ratio > 1.33:
            self.model.fp.rand_weight() # re-randomize weight if time scale varies significantly
            print("[Neural Buffer] Re-randomizing weights since ratio difference is: ", (self.prev_t_ratio / self.curr_t_ratio).cpu().numpy())

        optimizer = AdamW([{'params': self.model.parameters()}], 
                            lr=self.lr, betas=(0.9, 0.99))
        
        consecutive_success = 0

        importance = self.compute_importance()

        # main training loop
        for i in range(self.N_iters):
            # get training X and Y
            with torch.no_grad():
                coord_sampled, u_sampled = self.sample(curr_u_x, curr_u_y, curr_u_z, init_u_x, \
                                                    init_u_y, init_u_z, self.mid_ts, importance) # sample from current time
            loss = self.loss(self.model(coord_sampled), u_sampled)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if loss < self.success_threshold:
                consecutive_success += 1
            else:
                consecutive_success = 0
            
            if i >= 50 and consecutive_success >= self.N_success:
                torch.cuda.synchronize()
                t = time.time()-time0
                print(f"[Neural Buffer] Early-terminated at iter: {i} Time used: {t}")
                break

            # decay learning rate
            new_lrate = self.lr * (0.1 ** (i / 1500))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate

            if i % 100 == 0:
                if i == 0:
                    torch.cuda.synchronize()
                    time0 = time.time()
                else:
                    torch.cuda.synchronize()    
                    t = time.time()-time0
                    print(f"[Neural Buffer] Iter: {i} Loss: {loss.item()} Time used: {t}")
            
        with torch.no_grad():
            self.model_prev.fp.reinit_like(self.model.fp) # reinit prev model to be exactly like current one
            self.model_prev.load_state_dict(self.model.state_dict())
            self.prev_t_ratio = self.curr_t_ratio # save the current time scale used
        
        return loss.detach().cpu().numpy()
    
    # coord [N, 3]
    # return [N, 1]
    def pred_u_x(self, coord, mid_t):
        coord_xyzt = torch.cat((coord, self.curr_t_ratio * mid_t * torch.ones(coord.shape[0], 1)), axis = -1)
        return 1./self.magnitude_scale * self.model.eval_x(coord_xyzt)

    # coord [N, 3]
    # return [N, 1]
    def pred_u_y(self, coord, mid_t):
        coord_xyzt = torch.cat((coord, self.curr_t_ratio * mid_t * torch.ones(coord.shape[0], 1)), axis = -1)
        return 1./self.magnitude_scale * self.model.eval_y(coord_xyzt)

    # coord [N, 3]
    # return [N, 1]
    def pred_u_z(self, coord, mid_t):
        coord_xyzt = torch.cat((coord, self.curr_t_ratio * mid_t * torch.ones(coord.shape[0], 1)), axis = -1)
        return 1./self.magnitude_scale * self.model.eval_z(coord_xyzt)
        