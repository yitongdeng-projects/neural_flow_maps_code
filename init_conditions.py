# 
from taichi_utils import *
import math

# w: vortex strength
# rad: radius of torus
# delta: thickness of torus
# c: ring center position
# unit_x, unit_y: the plane of the circle
@ti.kernel
def add_vortex_ring_and_smoke(w: float, rad: float, delta: float, c: ti.types.vector(3, float),
                unit_x: ti.types.vector(3, float), unit_y: ti.types.vector(3, float),
                pf: ti.template(), vf: ti.template(), smokef: ti.template(), color: ti.types.vector(3, float), num_samples: int):
    curve_length = (2 * math.pi * rad) / num_samples # each sample point has an associated length
    for i, j, k in vf:
        for l in range(num_samples):
            theta = l/num_samples * 2 * (math.pi)
            p_sampled = rad * (ti.cos(theta) * unit_x + ti.sin(theta) * unit_y) + c # position of the sample point
            p_diff = pf[i,j,k]-p_sampled
            r = p_diff.norm()
            w_vector = w * (-ti.sin(theta) * unit_x + ti.cos(theta) * unit_y)
            vf[i,j,k] += curve_length * (-1/(4 * math.pi * r ** 3) * (1-ti.exp(-(r/delta) ** 3))) * p_diff.cross(w_vector)
            smokef[i,j,k][3] += curve_length * (ti.exp(-(r/delta) ** 3))
    for i, j, k in smokef:
        if smokef[i,j,k][3] > 0.002:
            smokef[i,j,k][3] = 1.0
            smokef[i,j,k].xyz = color
        else:
            smokef[i,j,k] = ti.Vector([0.,0.,0.,0.])

# initialize four vorts
def init_four_vorts(X, u, smoke1, smoke2):
    smoke1.fill(0.)
    smoke2.fill(0.)
    x_offset = 0.16
    y_offset = 0.16
    size = 0.15
    cos45 = ti.cos(math.pi/4)
    add_vortex_ring_and_smoke(w = 2.e-2, rad = size, delta = 0.024, c = ti.Vector([0.5-x_offset,0.5-y_offset, 1]),
        unit_x = ti.Vector([-cos45,cos45,0.]).normalized(), unit_y = ti.Vector([0.,0.,1.]),
        pf = X, vf = u, smokef = smoke1, color = ti.Vector([1,0.8,0.7]), num_samples = 500)

    add_vortex_ring_and_smoke(w = -2.e-2, rad = size, delta = 0.024, c = ti.Vector([0.5+x_offset,0.5-y_offset, 1]),
        unit_x = ti.Vector([cos45,cos45,0.]).normalized(), unit_y = ti.Vector([0.,0.,1.]),
        pf = X, vf = u, smokef = smoke2, color = ti.Vector([1,0.8,0.7]), num_samples = 500)
    
    add_fields(smoke1, smoke2, smoke1, 1.0)
    smoke2.fill(0.)

    add_vortex_ring_and_smoke(w = 2.e-2, rad = size, delta = 0.024, c = ti.Vector([0.5-x_offset,0.5+y_offset, 1]),
        unit_x = ti.Vector([cos45,cos45,0.]).normalized(), unit_y = ti.Vector([0.,0.,1.]),
        pf = X, vf = u, smokef = smoke2, color = ti.Vector([1,0.8,0.7]), num_samples = 500)
    
    add_fields(smoke1, smoke2, smoke1, 1.0)
    smoke2.fill(0.)

    add_vortex_ring_and_smoke(w = -2.e-2, rad = size, delta = 0.024, c = ti.Vector([0.5+x_offset,0.5+y_offset, 1]),
        unit_x = ti.Vector([-cos45,cos45,0.]).normalized(), unit_y = ti.Vector([0.,0.,1.]),
        pf = X, vf = u, smokef = smoke2, color = ti.Vector([1,0.8,0.7]), num_samples = 500)

    add_fields(smoke1, smoke2, smoke1, 1.0)