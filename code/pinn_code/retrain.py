import time
import sys
import numpy as np
from scipy.integrate import solve_ivp
import deepxde as dxe

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
import torch
torch.set_default_device('cpu') 

sys.path.append(os.path.abspath("../pinn_code"))
from physics_utility import poisson_residual

geom = dxe.geometry.Interval(0, 10)
def pde(x, y):
    dphi_dr = dxe.grad.jacobian(y, x)
    d2phi_dr2 = dxe.grad.hessian(y, x)
    return poisson_residual(x, y, dphi_dr, d2phi_dr2)

def boundary_center(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)

bc = dxe.icbc.DirichletBC(geom, lambda x: -1.0, boundary_center)

data = dxe.data.PDE(geom, pde, [bc], num_domain=500, num_boundary=2)
data = dxe.data.PDE(geom, pde, [], num_domain=500, num_boundary=2)

net = dxe.nn.FNN([1] + [64] * 4 + [1], "tanh", "Glorot normal")

model = dxe.model.Model(data, net)

model.compile("adam", lr=0.001)
model.train(epochs=10000)

model.compile("L-BFGS")
model.train(epochs=2000)

model.save("/home/sachins/myproject/research_project/code/pinn_code/MODELS/best_model_v2.ckpt")