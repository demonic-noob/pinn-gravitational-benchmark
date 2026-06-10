import os
import sys
import time
import torch
import numpy as np
import pandas as pd
import deepxde as dxe
from scipy.integrate import solve_ivp
from physics_utility import poisson_residual

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
torch.set_default_device('cpu') 

sys.path.append(os.path.abspath("../pinn_code"))

geom = dxe.geometry.Interval(0, 10)
def pde(x, y):
    dphi_dr = dxe.grad.jacobian(y, x)
    d2phi_dr2 = dxe.grad.hessian(y, x)
    return poisson_residual(x, y, dphi_dr, d2phi_dr2)

data_eval = dxe.data.PDE(geom, pde, [], num_domain=100, num_boundary=2)
net = dxe.nn.FNN([1] + [64] * 4 + [1], "tanh", "Glorot normal")
model = dxe.model.Model(data_eval, net)
model.compile("adam", lr=0.001)

checkpoint_path = "/home/sachins/myproject/research_project/code/pinn_code/MODELS/best_model_v2.ckpt-10041.pt"
checkpoint = torch.load(checkpoint_path, map_location='cpu')
model.net.load_state_dict(checkpoint["model_state_dict"])

np.random.seed(42)
num_points = 500
r_test = np.sort(np.random.rand(num_points, 1) * 10, axis=0)

def base_rho(r): 
    return (3.0 / (4.0 * np.pi)) * (1.0 + r**2)**(-2.5)

def perturbed_rho(r): 
    return base_rho(r) + 0.5 * np.sin(3 * r)

def analytical_perfect(r): 
    return -1.0 / np.sqrt(r**2 + 1.0)

def get_numerical_sol(r_vals, rho_func):
    sol_vals = []
    phi_0 = -1.0 / np.sqrt(0.01**2 + 1.0)
    dphi_0 = 0.01 / (0.01**2 + 1.0)**1.5
    
    for r in r_vals:
        if r[0] <= 0.01:
            sol_vals.append(phi_0)
            continue
        def ode(r_in, z):
            dphi, d2phi = z
            return [d2phi, -2.0/r_in * d2phi + 4.0*np.pi * rho_func(r_in)]
        sol = solve_ivp(ode, [0.01, r[0]], [phi_0, dphi_0], method='RK45')
        sol_vals.append(sol.y[0, -1])
    return np.array(sol_vals).reshape(-1, 1)

start = time.time()
phi_pinn = model.predict(r_test)
pinn_time = time.time() - start

start = time.time(); phi_num_perf = get_numerical_sol(r_test, base_rho); t_num_perf = time.time() - start
start = time.time(); phi_num_pert = get_numerical_sol(r_test, perturbed_rho); t_num_pert = time.time() - start
start = time.time(); phi_ana_perf = analytical_perfect(r_test); t_ana_perf = time.time() - start

def calculate_metrics(pred, true):
    pred_centered = pred - np.mean(pred) + np.mean(true)
    mae = np.mean(np.abs(pred_centered - true))
    mape = np.mean(np.abs((pred_centered - true) / (true + 1e-10))) * 100
    r_corr = np.corrcoef(pred.flatten(), true.flatten())[0, 1]
    return mae, mape, r_corr

mae_pinn_perf, mape_pinn_perf, r_pinn_perf = calculate_metrics(phi_pinn, phi_num_perf)
mae_pinn_pert, mape_pinn_pert, r_pinn_pert = calculate_metrics(phi_pinn, phi_num_pert)
mae_ana_perf, mape_ana_perf, r_ana_perf = calculate_metrics(phi_ana_perf, phi_num_perf)

print(f"\n{'Method':<12} | {'System':<10} | {'Time (s)':<10} | {'Aligned MAE':<12} | {'MAPE (%)':<10} | {'Corr (r)'}")
print("-" * 75)
print(f"{'PINN':<12} | {'Perfect':<10} | {pinn_time:<10.5f} | {mae_pinn_perf:<12.6f} | {mape_pinn_perf:<10.2f}% | {r_pinn_perf:.4f}")
print(f"{'PINN':<12} | {'Perturbed':<10} | {pinn_time:<10.5f} | {mae_pinn_pert:<12.6f} | {mape_pinn_pert:<10.2f}% | {r_pinn_pert:.4f}")
print(f"{'Analytical':<12} | {'Perfect':<10} | {t_ana_perf:<10.5f} | {mae_ana_perf:<12.6f} | {mape_ana_perf:<10.2f}% | {r_ana_perf:.4f}")
print(f"{'Analytical':<12} | {'Perturbed':<10} | {'N/A':<10} | {'N/A':<12} | {'N/A':<10} | {'N/A'}")
print(f"{'Numerical':<12} | {'Perfect':<10} | {t_num_perf:<10.5f} | {'0.000000':<12} | {'0.00':<10}% | {1.0000:.4f}")
print(f"{'Numerical':<12} | {'Perturbed':<10} | {t_num_pert:<10.5f} | {'0.000000':<12} | {'0.00':<10}% | {1.0000:.4f}")

df = pd.DataFrame({
    "Method": ["PINN", "PINN", "Analytical", "Numerical", "Numerical"],
    "System": ["Perfect", "Perturbed", "Perfect", "Perfect", "Perturbed"],
    "Time": [pinn_time, pinn_time, t_ana_perf, t_num_perf, t_num_pert],
    "Aligned_MAE": [mae_pinn_perf, mae_pinn_pert, mae_ana_perf, 0.0, 0.0],
    "Structural_Correlation": [r_pinn_perf, r_pinn_pert, r_ana_perf, 1.0, 1.0]
})
df.to_csv("benchmark_results.csv", index=False)
print("\nMetrics compiled and saved to benchmark_results.csv")
