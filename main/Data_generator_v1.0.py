import os
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

def generate_stationary_data(num_frames, noise_level):
    x = np.random.normal(0, noise_level, size=num_frames) + 1000
    y = np.random.normal(0, noise_level, size=num_frames) + 1000
    return x, y

def generate_non_stationary_data(num_frames, noise_level, jump_probability):
    x = np.cumsum(np.random.normal(0, noise_level, size=num_frames)) + 1000
    y = np.cumsum(np.random.normal(0, noise_level, size=num_frames)) + 1000

    for i in range(num_frames):
        if np.random.rand() < jump_probability:
            x[i] += np.random.normal(0, noise_level * 5)
            y[i] += np.random.normal(0, noise_level * 5)

    return x, y

def create_synthetic_data(subfolder_path, num_frames, noise_level, jump_probability, stationary=True):
    if stationary:
        x, y = generate_stationary_data(num_frames, noise_level)
    else:
        x, y = generate_non_stationary_data(num_frames, noise_level, jump_probability)

    data = pd.DataFrame({
        'Frame': range(num_frames),
        'Time': np.linspace(0, num_frames / 30, num_frames),  # Assuming 30 FPS for time calculation
        'X': x,
        'Y': y
    })
    
    txt_file_path = os.path.join(subfolder_path, 'tracked_data.txt')
    data.to_csv(txt_file_path, sep=',', index=False)

def generate_directories(base_path, subfolder_count, num_frames, noise_level, jump_probability, stationary_ratio):
    for i in range(subfolder_count):
        subfolder_path = os.path.join(base_path, f"subject_{i+1}")
        os.makedirs(subfolder_path, exist_ok=True)

        stationary = np.random.rand() < stationary_ratio
        create_synthetic_data(subfolder_path, num_frames, noise_level, jump_probability, stationary)

def browse_directory():
    folder_path = filedialog.askdirectory()
    if folder_path:
        folder_path_entry.delete(0, tk.END)
        folder_path_entry.insert(0, folder_path)

def start_generation():
    folder_path = folder_path_entry.get()
    if not folder_path:
        messagebox.showerror("Error", "Please select a folder path.")
        return

    try:
        subfolder_count = int(subfolder_count_entry.get())
        num_frames = int(num_frames_entry.get())
        noise_level = float(noise_level_entry.get())
        jump_probability = float(jump_probability_entry.get())
        stationary_ratio = float(stationary_ratio_entry.get())
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numeric values.")
        return

    generate_directories(folder_path, subfolder_count, num_frames, noise_level, jump_probability, stationary_ratio)
    messagebox.showinfo("Success", "Synthetic data generation complete.")

# GUI setup
root = tk.Tk()
root.title("Synthetic Data Generator")

tk.Label(root, text="Folder Path:").grid(row=0, column=0, padx=10, pady=10, sticky='e')
folder_path_entry = ttk.Entry(root, width=40)
folder_path_entry.grid(row=0, column=1, padx=10, pady=10)
browse_button = tk.Button(root, text="Browse", command=browse_directory)
browse_button.grid(row=0, column=2, padx=10, pady=10)

tk.Label(root, text="Subfolder Count:").grid(row=1, column=0, padx=10, pady=10, sticky='e')
subfolder_count_entry = ttk.Entry(root)
subfolder_count_entry.grid(row=1, column=1, padx=10, pady=10)

tk.Label(root, text="Number of Frames:").grid(row=2, column=0, padx=10, pady=10, sticky='e')
num_frames_entry = ttk.Entry(root)
num_frames_entry.grid(row=2, column=1, padx=10, pady=10)

tk.Label(root, text="Noise Level:").grid(row=3, column=0, padx=10, pady=10, sticky='e')
noise_level_entry = ttk.Entry(root)
noise_level_entry.grid(row=3, column=1, padx=10, pady=10)

tk.Label(root, text="Jump Probability:").grid(row=4, column=0, padx=10, pady=10, sticky='e')
jump_probability_entry = ttk.Entry(root)
jump_probability_entry.grid(row=4, column=1, padx=10, pady=10)

tk.Label(root, text="Stationary Ratio (0-1):").grid(row=5, column=0, padx=10, pady=10, sticky='e')
stationary_ratio_entry = ttk.Entry(root)
stationary_ratio_entry.grid(row=5, column=1, padx=10, pady=10)

generate_button = tk.Button(root, text="Generate Data", command=start_generation)
generate_button.grid(row=6, columnspan=3, pady=20)

root.mainloop()
