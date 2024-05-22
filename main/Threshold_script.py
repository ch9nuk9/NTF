import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import tkinter as tk
from tkinter import filedialog

def load_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        process_file(file_path)

def process_file(file_path):
    data = pd.read_csv(file_path, delim_whitespace=True, names=['frame', 'time', 'X', 'Y'])
    data['dX'] = data['X'].diff().fillna(0)
    data['dY'] = data['Y'].diff().fillna(0)
    data['distance'] = np.sqrt(data['dX']**2 + data['dY']**2)
    
    optimal_threshold = find_optimal_threshold(data)
    print(f"Optimal Threshold: {optimal_threshold}")

def find_optimal_threshold(data):
    distances = data['distance'].values
    density = gaussian_kde(distances)
    xs = np.linspace(0, np.max(distances), 1000)
    density_values = density(xs)
    
    threshold = xs[np.argmax(density_values > density_values.max() * 0.05)]
    
    plt.figure(figsize=(10, 6))
    plt.plot(xs, density_values, label='Density')
    plt.axvline(threshold, color='r', linestyle='--', label=f'Optimal Threshold: {threshold:.2f}')
    plt.xlabel('Distance')
    plt.ylabel('Density')
    plt.title('Optimal Threshold Determination')
    plt.legend()
    plt.show()
    
    return threshold

# Create the main window
root = tk.Tk()
root.title("Threshold Refinement")

# Create a button to load the file
load_button = tk.Button(root, text="Load File", command=load_file)
load_button.pack(pady=20)

# Run the application
root.mainloop()
