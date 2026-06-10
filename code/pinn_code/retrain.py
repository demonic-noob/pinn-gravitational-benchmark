import os
import sys
import torch
import deepxde as dxe
from physics_utility import poisson_residual

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
torch.set_default_device('cpu') 

sys.path.append(os.path.abspath("../pinn_code"))

geom = dxe.geometry.Interval(0, 10)

def pde(x, y):
    dphi_dr = dxe.grad.jacobian(y, x)
    d2phi_dr2 = dxe.grad.hessian(y, x)
    return poisson_residual(x, y, dphi_dr, d2phi_dr2)

def boundary_center(x, on_boundary):
    return on_boundary and dxe.utils.isclose(x[0], 0.0)

bc = dxe.icbc.DirichletBC(geom, lambda x: -1.0, boundary_center)

data = dxe.data.PDE(geom, pde, [bc], num_domain=500, num_boundary=2)

net = dxe.nn.FNN([1] + [64] * 4 + [1], "tanh", "Glorot normal")
model = dxe.model.Model(data, net)

print("--- Launching Phase 1: Adam Optimization ---")
model.compile("adam", lr=0.001)
model.train(epochs=10000)

print("--- Launching Phase 2: L-BFGS Fine-Tuning ---")
model.compile("L-BFGS")
model.train(epochs=2000)

output_path = "/home/sachins/myproject/research_project/code/pinn_code/MODELS/best_model_v2.ckpt"
model.save(output_path)
print(f"Successfully saved optimized model checkpoint to: {output_path}")
