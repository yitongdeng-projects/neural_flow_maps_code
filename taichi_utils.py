# 
import taichi as ti
import torch

eps = 1.e-6
data_type = ti.f32
torch_type = torch.float32

@ti.kernel
def copy_to(source: ti.template(), dest: ti.template()):
    for I in ti.grouped(source):
        dest[I] = source[I]

@ti.kernel
def scale_field(a: ti.template(), alpha: float, result: ti.template()):
    for I in ti.grouped(result):
        result[I] = alpha * a[I]

@ti.kernel
def add_fields(f1: ti.template(), f2: ti.template(), dest: ti.template(), multiplier: float):
    for I in ti.grouped(dest):
        dest[I] = f1[I] + multiplier * f2[I]

@ti.func
def lerp(vl, vr, frac):
    return vl + frac * (vr - vl)

@ti.kernel
def center_coords_func(pf: ti.template(), dx: float):
    for I in ti.grouped(pf):
        pf[I] = (I+0.5) * dx

@ti.kernel
def x_coords_func(pf: ti.template(), dx: float):
    for i, j, k in pf:
        pf[i, j, k] = ti.Vector([i, j + 0.5, k + 0.5]) * dx

@ti.kernel
def y_coords_func(pf: ti.template(), dx: float):
    for i, j, k in pf:
        pf[i, j, k] = ti.Vector([i + 0.5, j, k + 0.5]) * dx

@ti.kernel
def z_coords_func(pf: ti.template(), dx: float):
    for i, j, k in pf:
        pf[i, j, k] = ti.Vector([i + 0.5, j + 0.5, k]) * dx

@ti.func
def sample(qf: ti.template(), u: float, v: float, w: float):
    u_dim, v_dim, w_dim = qf.shape
    i = ti.max(0, ti.min(int(u), u_dim-1))
    j = ti.max(0, ti.min(int(v), v_dim-1))
    k = ti.max(0, ti.min(int(w), w_dim-1))
    return qf[i, j, k]

@ti.kernel
def curl(vf: ti.template(), cf: ti.template(), dx: float):
    inv_dist = 1./(2*dx)
    for i, j, k in cf:
        vr = sample(vf, i+1, j, k)
        vl = sample(vf, i-1, j, k)
        vt = sample(vf, i, j+1, k)
        vb = sample(vf, i, j-1, k)
        vc = sample(vf, i, j, k+1)
        va = sample(vf, i, j, k-1)

        d_vx_dz = inv_dist * (vc.x - va.x)
        d_vx_dy = inv_dist * (vt.x - vb.x)
        
        d_vy_dx = inv_dist * (vr.y - vl.y)
        d_vy_dz = inv_dist * (vc.y - va.y)

        d_vz_dx = inv_dist * (vr.z - vl.z)
        d_vz_dy = inv_dist * (vt.z - vb.z)

        cf[i,j,k][0] = d_vz_dy - d_vy_dz
        cf[i,j,k][1] = d_vx_dz - d_vz_dx
        cf[i,j,k][2] = d_vy_dx - d_vx_dy

@ti.kernel
def get_central_vector(vx: ti.template(), vy: ti.template(), vz: ti.template(), vc: ti.template()):
    for i, j, k in vc:
        vc[i,j,k].x = 0.5 * (vx[i+1, j, k] + vx[i, j, k])
        vc[i,j,k].y = 0.5 * (vy[i, j+1, k] + vy[i, j, k])
        vc[i,j,k].z = 0.5 * (vz[i, j, k+1] + vz[i, j, k])

@ti.kernel
def split_central_vector(vc: ti.template(), vx: ti.template(), vy: ti.template(), vz: ti.template()):
    for i, j, k in vx:
        r = sample(vc, i, j, k)
        l = sample(vc, i-1, j, k)
        vx[i,j,k] = 0.5 * (r.x + l.x)
    for i, j, k in vy:
        t = sample(vc, i, j, k)
        b = sample(vc, i, j-1, k)
        vy[i,j,k] = 0.5 * (t.y + b.y)
    for i, j, k in vz:
        c = sample(vc, i, j, k)
        a = sample(vc, i, j, k-1)
        vz[i,j,k] = 0.5 * (c.z + a.z)

@ti.kernel
def sizing_function(u: ti.template(), sizing: ti.template(), dx: float):
    u_dim, v_dim, w_dim = u.shape
    for i, j, k in u:
        u_l = sample(u, i-1, j, k)
        u_r = sample(u, i+1, j, k)
        u_t = sample(u, i, j+1, k)
        u_b = sample(u, i, j-1, k)
        u_c = sample(u, i, j, k+1)
        u_a = sample(u, i, j, k-1)
        partial_x = 1./(2*dx) * (u_r - u_l)
        partial_y = 1./(2*dx) * (u_t - u_b)
        partial_z = 1./(2*dx) * (u_c - u_a)
        if i == 0:
            partial_x = 1./(2*dx) * (u_r - 0)
        elif i == u_dim - 1:
            partial_x = 1./(2*dx) * (0 - u_l)
        if j == 0:
            partial_y = 1./(2*dx) * (u_t - 0)
        elif j == v_dim - 1:
            partial_y = 1./(2*dx) * (0 - u_b)
        if k == 0:
            partial_z = 1./(2*dx) * (u_c - 0)
        elif k == w_dim - 1:
            partial_z = 1./(2*dx) * (0 - u_a)

        sizing[i, j, k] = ti.sqrt(partial_x.x ** 2 + partial_x.y ** 2 + partial_x.z ** 2\
                            + partial_y.x ** 2 + partial_y.y ** 2 + partial_y.z ** 2\
                            + partial_z.x ** 2 + partial_z.y ** 2 + partial_z.z ** 2)

@ti.kernel
def diffuse_grid(value: ti.template(), tmp: ti.template()):
    for I in ti.grouped(value):
        value[I] = ti.abs(value[I])
    for i, j, k in tmp:
        tmp[i,j,k] = 1./6 * (sample(value, i+1,j,k) + sample(value, i-1,j,k)\
                + sample(value, i,j+1,k) + sample(value, i,j-1,k)\
                + sample(value, i,j,k+1) + sample(value, i,j,k-1))
    for I in ti.grouped(tmp):
        value[I] = ti.max(value[I], tmp[I])

# interpolation kernels

# linear
@ti.func
def N_1(x):
    return 1.0-ti.abs(x)
    
@ti.func
def dN_1(x):
    result = -1.0
    if x < 0.:
        result = 1.0
    return result

@ti.func
def interp_grad_1(vf, p, dx, BL_x = 0.5, BL_y = 0.5, BL_z = 0.5):
    u_dim, v_dim, w_dim = vf.shape

    u, v, w = p / dx
    u = u - BL_x
    v = v - BL_y
    w = w - BL_z
    s = ti.max(1., ti.min(u, u_dim-1-eps))
    t = ti.max(1., ti.min(v, v_dim-1-eps))
    l = ti.max(1., ti.min(w, w_dim-1-eps))

    # floor
    iu, iv, iw = ti.floor(s), ti.floor(t), ti.floor(l)

    partial_x = 0.
    partial_y = 0.
    partial_z = 0.
    interped = 0.

    # loop over indices
    for i in range(0, 2):
        for j in range(0, 2):
            for k in range(0, 2):
                x_p_x_i = s - (iu + i) # x_p - x_i
                y_p_y_i = t - (iv + j)
                z_p_z_i = l - (iw + k)
                value = sample(vf, iu + i, iv + j, iw + k)
                partial_x += 1./dx * (value * dN_1(x_p_x_i) * N_1(y_p_y_i) * N_1(z_p_z_i))
                partial_y += 1./dx * (value * N_1(x_p_x_i) * dN_1(y_p_y_i) * N_1(z_p_z_i))
                partial_z += 1./dx * (value * N_1(x_p_x_i) * N_1(y_p_y_i) * dN_1(z_p_z_i))
                interped += value * N_1(x_p_x_i) * N_1(y_p_y_i) * N_1(z_p_z_i)  
    
    return interped, ti.Vector([partial_x, partial_y, partial_z])

@ti.func
def interp_1(vf, p, dx, BL_x = 0.5, BL_y = 0.5, BL_z = 0.5):
    u_dim, v_dim, w_dim = vf.shape

    u, v, w = p / dx
    u = u - BL_x
    v = v - BL_y
    w = w - BL_z
    s = ti.max(1., ti.min(u, u_dim-1-eps))
    t = ti.max(1., ti.min(v, v_dim-1-eps))
    l = ti.max(1., ti.min(w, w_dim-1-eps))

    # floor
    iu, iv, iw = ti.floor(s), ti.floor(t), ti.floor(l)

    interped = 0. * sample(vf, iu, iv, iw)

    # loop over indices
    for i in range(0, 2):
        for j in range(0, 2):
            for k in range(0, 2):
                x_p_x_i = s - (iu + i) # x_p - x_i
                y_p_y_i = t - (iv + j)
                z_p_z_i = l - (iw + k)
                value = sample(vf, iu + i, iv + j, iw + k)
                interped += value * N_1(x_p_x_i) * N_1(y_p_y_i) * N_1(z_p_z_i)  
    
    return interped

@ti.func
def sample_min_max_1(vf, p, dx, BL_x = 0.5, BL_y = 0.5, BL_z = 0.5):
    u_dim, v_dim, w_dim = vf.shape

    u, v, w = p / dx
    u = u - BL_x
    v = v - BL_y
    w = w - BL_z
    s = ti.max(1., ti.min(u, u_dim-1-eps))
    t = ti.max(1., ti.min(v, v_dim-1-eps))
    l = ti.max(1., ti.min(w, w_dim-1-eps))

    # floor
    iu, iv, iw = ti.floor(s), ti.floor(t), ti.floor(l)

    mini = sample(vf, iu, iv, iw)
    maxi = sample(vf, iu, iv, iw)

    # loop over indices
    for i in range(0, 2):
        for j in range(0, 2):
            for k in range(0, 2):
                value = sample(vf, iu + i, iv + j, iw + k)
                mini = ti.min(mini, value)
                maxi = ti.max(maxi, value)

    return mini, maxi


# quadratic
@ti.func
def N_2(x):
    result = 0.0
    abs_x = ti.abs(x)
    if abs_x < 0.5:
        result = 3.0/4.0 - abs_x ** 2
    elif abs_x < 1.5:
        result = 0.5 * (3.0/2.0-abs_x) ** 2
    return result
    
@ti.func
def dN_2(x):
    result = 0.0
    abs_x = ti.abs(x)
    if abs_x < 0.5:
        result = -2 * abs_x
    elif abs_x < 1.5:
        result = 0.5 * (2 * abs_x - 3)
    if x < 0.0: # if x < 0 then abs_x is -x
        result *= -1
    return result

@ti.func
def interp_grad_2(vf, p, dx, BL_x = 0.5, BL_y = 0.5, BL_z = 0.5):
    u_dim, v_dim, w_dim = vf.shape

    u, v, w = p / dx
    u = u - BL_x
    v = v - BL_y
    w = w - BL_z
    s = ti.max(1., ti.min(u, u_dim-2-eps))
    t = ti.max(1., ti.min(v, v_dim-2-eps))
    l = ti.max(1., ti.min(w, w_dim-2-eps))

    # floor
    iu, iv, iw = ti.floor(s), ti.floor(t), ti.floor(l)

    partial_x = 0.
    partial_y = 0.
    partial_z = 0.
    interped = 0.

    # loop over indices
    for i in range(-1, 3):
        for j in range(-1, 3):
            for k in range(-1, 3):
                x_p_x_i = s - (iu + i) # x_p - x_i
                y_p_y_i = t - (iv + j)
                z_p_z_i = l - (iw + k)
                value = sample(vf, iu + i, iv + j, iw + k)
                partial_x += 1./dx * (value * dN_2(x_p_x_i) * N_2(y_p_y_i) * N_2(z_p_z_i))
                partial_y += 1./dx * (value * N_2(x_p_x_i) * dN_2(y_p_y_i) * N_2(z_p_z_i))
                partial_z += 1./dx * (value * N_2(x_p_x_i) * N_2(y_p_y_i) * dN_2(z_p_z_i))
                interped += value * N_2(x_p_x_i) * N_2(y_p_y_i) * N_2(z_p_z_i)  
    
    return interped, ti.Vector([partial_x, partial_y, partial_z])


# ti and torch conversion

@ti.kernel
def ti2torch(field: ti.template(), data: ti.types.ndarray()):
    for I in ti.grouped(data):
        data[I] = field[I]

@ti.kernel
def torch2ti(data: ti.types.ndarray(), field: ti.template()):
    for I in ti.grouped(data):
        field[I] = data[I]

@ti.kernel
def torch2ti_grad(grad: ti.types.ndarray(), field: ti.template()):
    for I in ti.grouped(grad):
        field.grad[I] = grad[I]

@ti.kernel
def ti2torch_grad(field: ti.template(), grad: ti.types.ndarray()):
    for I in ti.grouped(grad):
        grad[I] = field.grad[I]

#

@ti.kernel
def random_initialize(data: ti.types.ndarray()):
    for I in ti.grouped(data):
        data[I] = (ti.random() * 2.0 - 1.0) * 1e-4