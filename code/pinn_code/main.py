import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" 

import deepxde as dxe
import numpy as np
from physics_utility import poisson_residual
import matplotlib.pyplot as plt

geom = dxe.geometry.Interval(0, 10)

def pde(x, y):
    dphi_dr = dxe.grad.jacobian(y, x)
    d2phi_dr2 = dxe.grad.hessian(y, x)
    return poisson_residual(x, y, dphi_dr, d2phi_dr2)

def boundary_center(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)

bc = dxe.icbc.DirichletBC(geom, lambda x: -1.0, boundary_center)

data = dxe.data.PDE(geom, pde, [bc], num_domain=100, num_boundary=2)
net = dxe.nn.FNN([1] + [32] * 3 + [1], "tanh", "Glorot normal")
model = dxe.model.Model(data, net)
model.compile("adam", lr=0.001)

checkpointer = dxe.callbacks.ModelCheckpoint(
    "best_model.ckpt", 
    save_better_only=True, 
    period=1000 
)

model.train(iterations=10000, callbacks=[checkpointer])

