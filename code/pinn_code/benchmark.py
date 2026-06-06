import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import torch
torch.set_default_device('cpu') 

import time
import sys
import numpy as np
from scipy.integrate import solve_ivp
import deepxde as dxe

sys.path.append(os.path.abspath("../pinn_code"))
from physics_utility import poisson_residual
os.environ["CUDA_VISIBLE_DEVICES"] = ""

geom = dxe.geometry.Interval(0, 10)

def pde(x, y):
    dphi_dr = dxe.grad.jacobian(y, x)
    d2phi_dr2 = dxe.grad.hessian(y, x)
    return poisson_residual(x, y, dphi_dr, d2phi_dr2)

def boundary_center(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)

bc = dxe.icbc.DirichletBC(geom, lambda x: -1.0, boundary_center)
data = dxe.data.PDE(geom, pde, [bc], num_domain=100, num_boundary=2)
net = dxe.nn.FNN([1] + [64] * 4 + [1], "tanh", "Glorot normal")
model = dxe.model.Model(data, net)
model.compile("adam", lr=0.001)
checkpoint = torch.load("/home/sachins/myproject/research_project/code/pinn_code/MODELS/best_model_v2.ckpt-10043.pt", map_location='cpu')
model.net.load_state_dict(checkpoint["model_state_dict"])

num_points = 500
r_test = np.random.rand(num_points, 1) * 10

def base_rho(r): return 1.0 / (1 + r**2)**2.5
def perturbed_rho(r): return base_rho(r) + 0.5 * np.sin(3 * r)
def analytical_perfect(r): return -1.0 / np.sqrt(r**2 + 1.0)

def get_numerical_sol(r_vals, rho_func):
    sol_vals = []
    for r in r_vals:
        def ode(r_in, z):
            dphi, d2phi = z
            return [d2phi, -2/r_in * d2phi + 4*np.pi * rho_func(r_in)] if r_in > 0.01 else [d2phi, 0]
        sol = solve_ivp(ode, [0.01, r[0]], [0, 0], method='RK45')
        sol_vals.append(sol.y[0, -1])
    return np.array(sol_vals).reshape(-1, 1)

#PINN
start = time.time()
phi_pinn = model.predict(r_test)
pinn_time = time.time() - start

start = time.time(); phi_num_perf = get_numerical_sol(r_test, base_rho); t_num_perf = time.time() - start
start = time.time(); phi_num_pert = get_numerical_sol(r_test, perturbed_rho); t_num_pert = time.time() - start

start = time.time(); phi_ana_perf = analytical_perfect(r_test); t_ana_perf = time.time() - start

 
mae_pinn_perf = np.mean(np.abs(phi_pinn - phi_num_perf))
mae_pinn_pert = np.mean(np.abs(phi_pinn - phi_num_pert))
mae_ana_perf = np.mean(np.abs(phi_ana_perf - phi_num_perf))

print(f"\n{'Method':<12} | {'System':<10} | {'Time (s)':<10} | {'MAE (Accuracy)'}")
print("-" * 55)
print(f"{'PINN':<12} | {'Perfect':<10} | {pinn_time:<10.5f} | {mae_pinn_perf:.6f}")
print(f"{'PINN':<12} | {'Perturbed':<10} | {pinn_time:<10.5f} | {mae_pinn_pert:.6f}")
print(f"{'Analytical':<12} | {'Perfect':<10} | {t_ana_perf:<10.5f} | {mae_ana_perf:.6f}")
print(f"{'Analytical':<12} | {'Perturbed':<10} | {'N/A':<10} | {'N/A'}")
print(f"{'Numerical':<12} | {'Perfect':<10} | {t_num_perf:<10.5f} | {'0.000000'}")
print(f"{'Numerical':<12} | {'Perturbed':<10} | {t_num_pert:<10.5f} | {'0.000000'}")

import pandas as pd

data_report = {
    "Method": ["PINN", "PINN", "Analytical", "Numerical", "Numerical"],
    "System": ["Perfect", "Perturbed", "Perfect", "Perfect", "Perturbed"],
    "Time": [pinn_time, pinn_time, t_ana_perf, t_num_perf, t_num_pert],
    "MAE": [mae_pinn_perf, mae_pinn_pert, mae_ana_perf, 0.0, 0.0]
}

df = pd.DataFrame(data_report)
df.to_csv("benchmark_results.csv", index=False)
print("Report saved to benchmark_results.csv")