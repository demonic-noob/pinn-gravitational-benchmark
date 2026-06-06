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

model.restore("/home/sachins/myproject/research_project/code/pinn_code/MODELS/best_model.ckpt-4000.pt", verbose=1)
r_test = np.linspace(0, 10, 100).reshape(-1, 1)
phi_pred = model.predict(r_test)

plt.figure(figsize=(8, 5))
plt.plot(r_test, phi_pred, label="PINN Predicted Potential")
plt.xlabel("Radius (r)")
plt.ylabel("Potential $\Phi(r)$")
plt.legend()
plt.show()