# 
from hyperparameters import *
import numpy as np
import taichi as ti
import torch
from taichi_utils import *
from torch.cuda.amp import custom_bwd, custom_fwd


t_discrete = 4 # number of temporal discretizations
t_intv = 1./(t_discrete-1)
total_feat_dim = feat_dim * t_discrete


@ti.data_oriented
class Sparse3TEncoder(torch.nn.Module):

    def __init__(self, max_num_queries=10000):

        super(Sparse3TEncoder, self).__init__()

        self.max_num_queries = max_num_queries

        self.activate_threshold = activate_threshold # a hyperparameter

        self.encode_kernel = self.interp_kernel_1

        self.res = [(min_res[0] * 2 ** l + 1, min_res[1] * 2 ** l + 1, min_res[2] * 2 ** l + 1) for l in range(num_levels)]

        self.count = ti.field(ti.i32, shape=())

        self.feats = [ti.field(data_type, needs_grad=True) for _ in range(num_levels)]
        self.serials = [ti.field(ti.i32) for _ in range(num_levels)] # serialized index
        self.blocks = [ti.root.bitmasked(ti.ijk, (int(ti.ceil(self.res[l][0]/4)), int(ti.ceil(self.res[l][1]/4)), int(ti.ceil(self.res[l][2]/4)))) for l in range(num_levels)] # this is sparse
        self.pixels = [self.blocks[l].dense(ti.ijk, (4, 4, 4)) for l in range(num_levels)]
        self.grad_blocks = [ti.root.bitmasked(ti.ijk, (int(ti.ceil(self.res[l][0]/4)), int(ti.ceil(self.res[l][1]/4)), int(ti.ceil(self.res[l][2]/4)))) for l in range(num_levels)] # this is sparse
        self.grad_pixels = [self.grad_blocks[l].dense(ti.ijk, (4, 4, 4)) for l in range(num_levels)]
        for l in range(num_levels):
            self.pixels[l].place(self.serials[l])
            self.pixels[l].dense(ti.l, total_feat_dim).place(self.feats[l])
            self.grad_pixels[l].dense(ti.l, total_feat_dim).place(self.feats[l].grad)

        # the taichi storage for output
        self.output = ti.field(dtype=data_type,
                                shape=(max_num_queries, feat_dim * num_levels),
                                needs_grad=True)

        # the taichi storage for input
        self.input = ti.field(dtype=data_type,
                                shape=(max_num_queries, 4), # input will have 4 coordinates
                                needs_grad=True)

        # torch buffer for output, should have the same size as self.output
        self.register_buffer(
            'output_torch',
            torch.zeros(max_num_queries, feat_dim * num_levels, dtype=torch_type),
            persistent=False,
        )

        class _module_function(torch.autograd.Function):

            @staticmethod
            @custom_fwd(cast_inputs=torch_type)
            def forward(ctx, input_pos, plane_params):
                # torch, slice of output buffer as needed
                output_torch = self.output_torch[:input_pos.shape[0]].contiguous()
                # copy input from torch to taichi
                torch2ti(input_pos.contiguous(), self.input)
                # copy params from torch to taichi
                for l in range(num_levels):
                    self.torch2grids(l, plane_params.contiguous())

                for l in range(num_levels):
                    self.encode_kernel(
                        self.input, # query pos
                        self.output, # queried output
                        l,
                        input_pos.shape[0], # number of queries
                    )
                # copy output from taichi to torch
                ti2torch(self.output, output_torch)

                return output_torch

            @staticmethod
            @custom_bwd
            def backward(ctx, doutput):

                self.zero_grad() # clear grad in taichi

                # copy the upstream gradient from torch to taichi
                torch2ti_grad(doutput.contiguous(), self.output)

                for l in range(num_levels):
                    # run grad of kernel
                    self.encode_kernel.grad(
                        self.input, # query pos
                        self.output, # queried output
                        l,
                        doutput.shape[0], # number of queries
                    )

                # copy downstream gradient from taichi to torch
                for l in range(num_levels):
                    self.grids2torch_grad(l, self.feats_torch_grad.contiguous())

                return None, self.feats_torch_grad

        self._module_function = _module_function

    def count_total_params(self):
        self.count[None] = 0
        for l in range(num_levels):
            self.count_layer_params(l)
        return self.count[None]
 
    @ti.kernel
    def count_layer_params(self, l: ti.template()):
        for i, j, k in self.serials[l]: # loops through all activated
            if self.serials[l][i,j,k] > 0: # count only if > 0
                ti.atomic_add(self.count[None], 1)

    def rand_weight(self):
        random_initialize(self.feats_torch)
        
    def reinit(self, sizing, verbose = False):
        for l in range(num_levels):
            self.serials[l].fill(0)
        # deactivate all
        self.deactivate()
        self.activate(sizing)
        if verbose:
            self.print_sparsity()
        total_num_params = self.count_total_params() # allocate this much for torch
        print("[Neural Buffer] Total num params: ", total_num_params, "x", total_feat_dim, "=", total_num_params * total_feat_dim)

        # allocated torch parameters to optimize
        self.feats_torch = torch.nn.Parameter(torch.zeros((total_num_params, total_feat_dim),
                                            dtype=torch_type),
                                            requires_grad=True)
        random_initialize(self.feats_torch) # randomly initialize

        # the grad for feats_torch
        self.register_buffer(
            'feats_torch_grad',
            torch.zeros((total_num_params, total_feat_dim), dtype=torch_type),
            persistent=False
        )

        print("[Neural Buffer] Encoder reinitialized.")

    # grow encoder
    def grow(self, sizing, verbose = False):
        self.activate_more(sizing)
        if verbose:
            self.print_sparsity()
        total_num_params = self.count_total_params() # allocate this much for torch
        print("[Neural Buffer] Total num params: ", total_num_params, "x", total_feat_dim, "=", total_num_params * total_feat_dim)

        # allocated torch parameters to optimize
        extended_feats_torch = torch.nn.Parameter(torch.zeros((total_num_params, total_feat_dim),
                                            dtype=torch_type),
                                            requires_grad=True)
        random_initialize(extended_feats_torch) # randomly initialize
        with torch.no_grad():
            extended_feats_torch[:self.feats_torch.shape[0], :] = self.feats_torch
        self.feats_torch = extended_feats_torch

        # the grad for feats_torch
        self.register_buffer(
            'feats_torch_grad',
            torch.zeros((total_num_params, total_feat_dim), dtype=torch_type),
            persistent=False
        )

        print("[Neural Buffer] Encoder grown.")

    # reinit like another instance
    def reinit_like(self, other, verbose = False):
        for l in range(num_levels):
            self.serials[l].fill(0)
        self.deactivate()
        self.activate_like(other)
        if verbose:
            self.print_sparsity()
        total_num_params = self.count_total_params() # allocate this much for torch
        print("[Neural Buffer] Total num params: ", total_num_params, "x", total_feat_dim, "=", total_num_params * total_feat_dim)

        # allocated torch parameters to optimize
        self.feats_torch = torch.nn.Parameter(torch.zeros((total_num_params, total_feat_dim),
                                            dtype=torch_type),
                                            requires_grad=True)
        random_initialize(self.feats_torch) # randomly initialize

        # the grad for feats_torch
        self.register_buffer(
            'feats_torch_grad',
            torch.zeros((total_num_params, total_feat_dim), dtype=torch_type),
            persistent=False
        )
        print("[Neural Buffer] Encoder reinitialized like another one.")

    def zero_grad(self):
        for l in range(num_levels):
            self.zero_layer_grad(l)

    @ti.kernel
    def zero_layer_grad(self, l: ti.template()):
        ti.loop_config(block_dim = 16)
        for i,j,k in self.grad_pixels[l]:
            for m in ti.static(range(total_feat_dim)):
                self.feats[l].grad[i,j,k,m] = 0.

    def forward(self, positions):
        return self._module_function.apply(positions, self.feats_torch)

    @ti.kernel
    def interp_kernel_1(self,
            xyzts: ti.template(), # this is the queried positions
            xyzts_embedding: ti.template(), # this is the interpolated values
            l: ti.template(),
            B: ti.i32, # the number of queries
            ):

        ti.loop_config(block_dim=16)
        for i in range(B):
            for j in ti.static(range(feat_dim)):
                xyzts_embedding[i, l * feat_dim + j] *= 0 # clear output first

        ti.loop_config(block_dim=32)
        for i, uvw in ti.ndrange(B, 8):
            xyz = ti.Vector([xyzts[i, 0], xyzts[i, 1], xyzts[i, 2]]) # position of i
            t = xyzts[i, 3]
            inv_dx = self.res[l][1] - 1
            pos = xyz * inv_dx # pos in local coordinate
            BL = ti.cast(ti.floor(pos), ti.int32) # floor
            pos -= BL # pos now represents frac
            BL_x = BL[0]
            BL_y = BL[1]
            BL_z = BL[2]

            u = uvw // 4
            v = (uvw % 4) // 2
            w = (uvw % 4) % 2

            xp_xi = ti.abs(pos[0] - u)
            yp_yi = ti.abs(pos[1] - v)
            zp_zi = ti.abs(pos[2] - w)

            t_idx = ti.Vector([0, 1, 2, 3])
            ts = t_idx * t_intv
            t1000 = ti.Vector([ts[1], ts[0], ts[0], ts[0]])
            t2211 = ti.Vector([ts[2], ts[2], ts[1], ts[1]])
            t3332 = ti.Vector([ts[3], ts[3], ts[3], ts[2]])
            V = ((t-t1000)*(t-t2211)*(t-t3332))/((ts-t1000)*(ts-t2211)*(ts-t3332))
            
            if (0 <= BL_x + u < self.res[l][0]) and (0 <= BL_y + v < self.res[l][1]) and (0 <= BL_z + w < self.res[l][2]):
                if self.serials[l][BL_x + u, BL_y + v, BL_z + w] > 0:
                    W = (1.-xp_xi) * (1.-yp_yi) * (1.-zp_zi)

                    for j in ti.static(range(feat_dim)):
                        a0 = self.feats[l][BL_x + u, BL_y + v, BL_z + w, j * t_discrete + t_idx[0]]
                        a1 = self.feats[l][BL_x + u, BL_y + v, BL_z + w, j * t_discrete + t_idx[1]]
                        a2 = self.feats[l][BL_x + u, BL_y + v, BL_z + w, j * t_discrete + t_idx[2]]
                        a3 = self.feats[l][BL_x + u, BL_y + v, BL_z + w, j * t_discrete + t_idx[3]]
                        xyzts_embedding[i, l * feat_dim + j] += (a0 * V[0] + a1 * V[1] + a2 * V[2] + a3 * V[3]) * W

    @ti.kernel
    def torch2grids(self, l: ti.template(), data: ti.types.ndarray()):
        ti.loop_config(block_dim=16)
        for i, j, k in self.pixels[l]: # loop only the activated
            if self.serials[l][i,j,k] > 0:
                serial = self.serials[l][i,j,k] - 1
                for m in ti.static(range(total_feat_dim)):
                    self.feats[l][i,j,k,m] = data[serial, m]

    @ti.kernel
    def grids2torch_grad(self, l: ti.template(), grad: ti.types.ndarray()):
        ti.loop_config(block_dim=16)
        for i, j, k in self.grad_pixels[l]: # loop only the activated
            if self.serials[l][i,j,k] > 0: # copy only if > 0
                serial = self.serials[l][i,j,k] - 1 # remember that serial started from 1 (0 means not activated)
                for m in ti.static(range(total_feat_dim)):
                    grad[serial, m] = self.feats[l].grad[i,j,k,m]

    def print_sparsity(self):
        print("\n==== Sparsity Printout ====")
        for l in range(num_levels):
            self.print_layer_sparsity(l)
        print("==== End Sparsity Printout ====\n")

    @ti.kernel
    def print_layer_sparsity(self, l: ti.template()):
        total_active = 0
        for i, j, k in self.blocks[l]: # only finds activated
            ti.atomic_add(total_active, 1)
        total_available = self.blocks[l].shape[0] * self.blocks[l].shape[1] * self.blocks[l].shape[2]

        print("-- Sparsity for Layer: ", l)
        print("Total sparsity: ", total_active/total_available)
        print("Total active blocks: ", total_active)
        print("Total active pixels: ", total_active * 4 * 4 * 4)

    def deactivate(self):
        for l in range(num_levels):
            self.blocks[l].deactivate_all()

    def activate(self, sizing):
        self.count[None] = 1 # start at 1, not 0 -- 0 is reserved for inactive pixels
        self.activate_entire_layer(0)
        if sizing is None:
            pass
        else:
            for l in range(1, num_levels):
                self.activate_layer(l, sizing)

    def activate_more(self, sizing):
        self.count[None] = 1 + self.feats_torch.shape[0] # start at 1 plus however many already activated
        for l in range(1, num_levels):
            self.activate_layer_more(l, sizing)

    def activate_like(self, other):
        for l in range(num_levels):
            self.activate_layer_like(l, other)

    # activate like another instance
    @ti.kernel
    def activate_layer_like(self, l: ti.template(), other: ti.template()):
        for i, j, k in other.serials[l]:
            self.serials[l][i,j,k] = other.serials[l][i,j,k]
            
    @ti.kernel
    def activate_entire_layer(self, l: ti.template()):
        for i, j, k in ti.ndrange(self.res[l][0], self.res[l][1], self.res[l][2]):
            self.serials[l][i,j,k] = ti.atomic_add(self.count[None], 1) # careful about racing

    # activate based on pre-computed sizing values
    @ti.kernel
    def activate_layer(self, l: ti.template(), sizing: ti.template()):
        u_dim, v_dim, w_dim = sizing.shape
        ratio_x = u_dim / (self.res[l][0] - 1)
        ratio_y = v_dim / (self.res[l][1] - 1)
        ratio_z = w_dim / (self.res[l][2] - 1)

        for i, j, k in ti.ndrange(self.res[l][0] - 1, self.res[l][1] - 1, self.res[l][2] - 1):
            max_sizing = 0.0
            x_begin = int(ti.floor(i * ratio_x))
            y_begin = int(ti.floor(j * ratio_y))
            z_begin = int(ti.floor(k * ratio_z))
            x_end = int(ti.ceil((i+1) * ratio_x))
            y_end = int(ti.ceil((j+1) * ratio_y))
            z_end = int(ti.ceil((k+1) * ratio_z))

            for x, y, z in ti.ndrange(x_end-x_begin, y_end-y_begin, z_end-z_begin):
                ti.atomic_max(max_sizing, ti.abs(sizing[x_begin + x, y_begin + y, z_begin + z]))

            if max_sizing > self.activate_threshold * (self.res[l][1] - 1):
                # temporally write to -1 to mark activity
                for x in ti.static(range(2)):
                    for y in ti.static(range(2)):
                        for z in ti.static(range(2)):
                            self.serials[l][i+x,j+y,k+z] = -1
        
        # loop again to assign actual serial
        for i, j, k in self.serials[l]:
            if self.serials[l][i,j,k] <= -1: # if marked before
                self.serials[l][i,j,k] = ti.atomic_add(self.count[None], 1) # careful about racing

    # activate more based on pre-computed sizing values
    @ti.kernel
    def activate_layer_more(self, l: ti.template(), sizing: ti.template()):
        u_dim, v_dim, w_dim = sizing.shape
        ratio_x = u_dim / (self.res[l][0] - 1)
        ratio_y = v_dim / (self.res[l][1] - 1)
        ratio_z = w_dim / (self.res[l][2] - 1)

        for i, j, k in ti.ndrange(self.res[l][0] - 1, self.res[l][1] - 1, self.res[l][2] - 1):
            max_sizing = 0.0
            x_begin = int(ti.floor(i * ratio_x))
            y_begin = int(ti.floor(j * ratio_y))
            z_begin = int(ti.floor(k * ratio_z))
            x_end = int(ti.ceil((i+1) * ratio_x))
            y_end = int(ti.ceil((j+1) * ratio_y))
            z_end = int(ti.ceil((k+1) * ratio_z))

            for x, y, z in ti.ndrange(x_end-x_begin, y_end-y_begin, z_end-z_begin):
                ti.atomic_max(max_sizing, ti.abs(sizing[x_begin + x, y_begin + y, z_begin + z]))

            if max_sizing > self.activate_threshold * (self.res[l][1] - 1):
                # temporally write to -1 to mark activity
                for x in ti.static(range(2)):
                    for y in ti.static(range(2)):
                        for z in ti.static(range(2)):
                            # only IF NOT ALREADY ACTIVATED, write -1
                            if self.serials[l][i+x,j+y,k+z] <= 0:
                                self.serials[l][i+x,j+y,k+z] = -1
        
        # loop again to assign actual serial
        for i, j, k in self.serials[l]:
            if self.serials[l][i,j,k] <= -1: # if marked before
                self.serials[l][i,j,k] = ti.atomic_add(self.count[None], 1) # careful about racing


    def paint_active(self, display):
        display.fill(0.)
        for l in range(num_levels):
            self.paint_layer_active(l, display)

    # draw grid mesh
    @ti.kernel
    def sketch_layer_active(self, l: ti.template(), display: ti.types.ndarray()):
        u_dim, v_dim = display.shape
        ratio_x = u_dim / (self.res[l][0]-1)
        ratio_y = v_dim / (self.res[l][1]-1)
        idx_z = self.res[l][2] // 2
        for i, j in display:
            idx_x = int(i // ratio_x) # this is the BL corner
            idx_y = int(j // ratio_y)
            # only count cell activated if all four corners are activated
            # if only some are activated then it doesn't count
            if self.serials[l][idx_x, idx_y, idx_z] > 0 and self.serials[l][idx_x+1, idx_y, idx_z] > 0 and\
                self.serials[l][idx_x, idx_y+1, idx_z] > 0 and self.serials[l][idx_x+1, idx_y+1, idx_z] > 0:
                display[i, j] += (1./num_levels) * 0.2
                if int((i-1) // ratio_x) != idx_x or int((i+1) // ratio_x) != idx_x or\
                    int((j-1) // ratio_y) != idx_y or int((j+1) // ratio_y) != idx_y:
                    display[i, j] = 0.0

    # draw black and white blocks
    @ti.kernel
    def paint_layer_active(self, l: ti.template(), display: ti.types.ndarray()):
        u_dim, v_dim = display.shape
        ratio_x = u_dim / (self.res[l][0]-1)
        ratio_y = v_dim / (self.res[l][1]-1)
        idx_z = self.res[l][2] // 2
        for i, j in display:
            idx_x = int(i // ratio_x) # this is the BL corner
            idx_y = int(j // ratio_y)
            # only count cell activated if all four corners are activated
            # if only some are activated then it doesn't count
            if self.serials[l][idx_x, idx_y, idx_z] > 0 and self.serials[l][idx_x+1, idx_y, idx_z] > 0 and\
                self.serials[l][idx_x, idx_y+1, idx_z] > 0 and self.serials[l][idx_x+1, idx_y+1, idx_z] > 0:
                if (idx_x+idx_y) % 2 == 0:
                    display[i, j] = 0.0
                else:
                    display[i, j] = 1

