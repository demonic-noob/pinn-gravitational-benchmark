import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

df = pd.read_csv('raw_data.csv')

r = df['r'].values
true_phi = df['numerical_phi'].values
pinn_phi = df['pinn_phi'].values

def plot_spectral_bias(r, true_phi, pinn_phi):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(r, true_phi, color='black', linestyle='--')
    ax1.plot(r, pinn_phi, color='red', alpha=0.7)
    ax1.set_title("Gravitational Potential (Spatial)")
    ax1.set_xlabel("Radius (r)")
    
    freq_true = np.abs(np.fft.rfft(true_phi))
    freq_pinn = np.abs(np.fft.rfft(pinn_phi))
    ax2.plot(freq_true, color='black')
    ax2.plot(freq_pinn, color='red')
    ax2.set_title("Frequency Domain (Spectral Bias)")
    ax2.set_xlabel("Frequency Index")
    ax2.set_xlim(0, 50) 
    
    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spectral_analysis.png')
    plt.savefig(save_path, dpi=300)
    plt.show()

plot_spectral_bias(r, true_phi, pinn_phi)