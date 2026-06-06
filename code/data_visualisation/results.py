import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("benchmark_results.csv")

df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
df['MAE'] = pd.to_numeric(df['MAE'], errors='coerce')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

labels = [f"{m}\n({s})" for m, s in zip(df['Method'], df['System'])]

ax1.bar(labels, df['Time'], color=['skyblue', 'skyblue', 'lightgreen', 'orange', 'orange', 'orange'])
ax1.set_yscale('log')
ax1.set_ylabel("Inference Time (s) [Log Scale]")
ax1.set_title("Performance Comparison (All Methods)")
ax1.grid(axis='y', linestyle='--', alpha=0.5)

ax2.bar(labels, df['MAE'], color=['skyblue', 'skyblue', 'lightgreen', 'orange', 'orange', 'orange'])
ax2.set_ylabel("Mean Absolute Error (MAE)")
ax2.set_title("Accuracy Comparison (All Methods)")
ax2.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("thesis_plots_full.png", dpi=300)
plt.show()