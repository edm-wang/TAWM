import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dts = [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.02, 0.03, 0.05]
    mts3_time = [2.45, 2.48, 4.86, 6.23, 10.55, 18.76, 56.66, 98.75, ]
    ta_time = [0.062, 0.063, 0.063, 0.067, 0.063, 0.068, 0.065, 0.058]

    plt.figure(figsize=(8, 5))  # Set figure size

    # print(df.groupby('eval_dt')['time'])
    plt.plot(dts, mts3_time, marker='o', linestyle='-', color='darkblue', label='MTS3 Model runtime')  # Plot data
    plt.plot(dts, ta_time, marker='o', linestyle='-', color='red', label='Time-Aware Model runtime')  # Plot data
    plt.xlabel('Evaluation $\Delta t$', fontsize=16)  # Label x-axis
    plt.ylabel('Runtime (seconds)', fontsize=16)  # Label y-axis
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5)  # Add grid
    plt.legend(fontsize=20)  # Add legend
    plt.show()  # Show plot
    plt.savefig('mts3-runtime.png')
    