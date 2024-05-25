import os
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import matplotlib.pyplot as plt
import logging
from typing import Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_stationary_data(num_frames: int, noise_level: float) -> Tuple[np.ndarray, np.ndarray]:
    x = np.random.normal(0, noise_level, size=num_frames) + 1000
    y = np.random.normal(0, noise_level, size=num_frames) + 1000
    return x, y

def generate_non_stationary_data(num_frames: int, noise_level: float, jump_probability: float) -> Tuple[np.ndarray, np.ndarray]:
    x = np.cumsum(np.random.normal(0, noise_level, size=num_frames)) + 1000
    y = np.cumsum(np.random.normal(0, noise_level, size=num_frames)) + 1000

    for i in range(num_frames):
        if np.random.rand() < jump_probability:
            x[i] += np.random.normal(0, noise_level * 5)
            y[i] += np.random.normal(0, noise_level * 5)

    return x, y

def create_synthetic_data(subfolder_path: str, num_frames: int, noise_level: float, jump_probability: float, stationary: bool = True) -> None:
    try:
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
        data.to_csv(txt_file_path, sep=' ', index=False)  # Use space delimiter to match expected format
        logging.info(f'Data successfully saved to {txt_file_path}')
    except Exception as e:
        logging.error(f'Error in creating synthetic data: {e}')
        raise

def generate_directories(base_path: str, subfolder_count: int, num_frames: int, noise_level: float, jump_probability: float, stationary_ratio: float) -> None:
    try:
        for i in range(subfolder_count):
            subfolder_path = os.path.join(base_path, f"subject_{i+1}")
            os.makedirs(subfolder_path, exist_ok=True)

            stationary = np.random.rand() < stationary_ratio
            create_synthetic_data(subfolder_path, num_frames, noise_level, jump_probability, stationary)
    except Exception as e:
        logging.error(f'Error in generating directories: {e}')
        raise

def browse_directory() -> None:
    folder_path = filedialog.askdirectory()
    if folder_path:
        folder_path_entry.delete(0, tk.END)
        folder_path_entry.insert(0, folder_path)

def start_generation_threaded() -> None:
    thread = threading.Thread(target=start_generation)
    thread.start()

def start_generation() -> None:
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
        
        # Validate stationary ratio
        if not (0 <= stationary_ratio <= 1):
            raise ValueError("Stationary ratio must be between 0 and 1.")
    except ValueError as e:
        messagebox.showerror("Error", f"Invalid input: {e}")
        return

    # Show progress bar
    progress_bar['value'] = 0
    progress_label['text'] = 'Generating data...'
    root.update_idletasks()

    try:
        generate_directories(folder_path, subfolder_count, num_frames, noise_level, jump_probability, stationary_ratio)
        progress_bar['value'] = 100
        progress_label['text'] = 'Generation complete'
        messagebox.showinfo("Success", "Synthetic data generation complete.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during generation: {e}")

def show_example_data() -> None:
    try:
        num_frames = int(num_frames_entry.get())
        noise_level = float(noise_level_entry.get())
        jump_probability = float(jump_probability_entry.get())
        stationary = example_type.get() == 'Stationary'
        
        if stationary:
            x, y = generate_stationary_data(num_frames, noise_level)
        else:
            x, y = generate_non_stationary_data(num_frames, noise_level, jump_probability)
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, marker='o', linestyle='-', markersize=2)
        plt.title('Example Data')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.show()
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numeric values.")

# GUI setup
root = tk.Tk()
root.title("Synthetic Data Generator")

def add_tooltip(widget, text):
    tool_tip = tk.Toplevel(widget)
    tool_tip.wm_overrideredirect(True)
    tool_tip.withdraw()
    tool_tip_label = ttk.Label(tool_tip, text=text, background="lightyellow", borderwidth=1, relief="solid")
    tool_tip_label.pack()

    def enter(event):
        tool_tip.wm_deiconify()
        tool_tip.geometry(f"+{widget.winfo_rootx() + 20}+{widget.winfo_rooty() + 20}")

    def leave(event):
        tool_tip.withdraw()

    widget.bind("<Enter>", enter)
    widget.bind("<Leave>", leave)

tk.Label(root, text="Folder Path:").grid(row=0, column=0, padx=10, pady=10, sticky='e')
folder_path_entry = ttk.Entry(root, width=40)
folder_path_entry.grid(row=0, column=1, padx=10, pady=10)
browse_button = tk.Button(root, text="Browse", command=browse_directory)
browse_button.grid(row=0, column=2, padx=10, pady=10)
add_tooltip(browse_button, "Browse to select the base folder path.")

tk.Label(root, text="Subfolder Count:").grid(row=1, column=0, padx=10, pady=10, sticky='e')
subfolder_count_entry = ttk.Entry(root)
subfolder_count_entry.grid(row=1, column=1, padx=10, pady=10)
subfolder_count_entry.insert(0, "10")
add_tooltip(subfolder_count_entry, "Enter the number of subfolders to create.")

tk.Label(root, text="Number of Frames:").grid(row=2, column=0, padx=10, pady=10, sticky='e')
num_frames_entry = ttk.Entry(root)
num_frames_entry.grid(row=2, column=1, padx=10, pady=10)
num_frames_entry.insert(0, "1000")
add_tooltip(num_frames_entry, "Enter the number of frames (data points) to generate.")

tk.Label(root, text="Noise Level:").grid(row=3, column=0, padx=10, pady=10, sticky='e')
noise_level_entry = ttk.Entry(root)
noise_level_entry.grid(row=3, column=1, padx=10, pady=10)
noise_level_entry.insert(0, "1.0")
add_tooltip(noise_level_entry, "Enter the noise level for data generation.")

tk.Label(root, text="Jump Probability:").grid(row=4, column=0, padx=10, pady=10, sticky='e')
jump_probability_entry = ttk.Entry(root)
jump_probability_entry.grid(row=4, column=1, padx=10, pady=10)
jump_probability_entry.insert(0, "0.05")
add_tooltip(jump_probability_entry, "Enter the probability of a jump occurring in non-stationary data.")

tk.Label(root, text="Stationary Ratio (0-1):").grid(row=5, column=0, padx=10, pady=10, sticky='e')
stationary_ratio_entry = ttk.Entry(root)
stationary_ratio_entry.grid(row=5, column=1, padx=10, pady=10)
stationary_ratio_entry.insert(0, "0.5")
add_tooltip(stationary_ratio_entry, "Enter the ratio of stationary to non-stationary data (0 to 1).")

generate_button = tk.Button(root, text="Generate Data", command=start_generation_threaded)
generate_button.grid(row=6, column=0, columnspan=2, pady=20)
add_tooltip(generate_button, "Click to start the data generation process.")

example_type = tk.StringVar(value='Stationary')
tk.Radiobutton(root, text="Stationary", variable=example_type, value='Stationary').grid(row=7, column=0, padx=10, pady=5)
tk.Radiobutton(root, text="Non-Stationary", variable=example_type, value='Non-Stationary').grid(row=7, column=1, padx=10, pady=5)
example_button = tk.Button(root, text="Show Example Data", command=show_example_data)
example_button.grid(row=7, column=2, padx=10, pady=5)
add_tooltip(example_button, "Click to display an example plot of the data.")

progress_frame = tk.Frame(root)
progress_frame.grid(row=8, column=0, columnspan=3, pady=10)
progress_label = tk.Label(progress_frame, text="")
progress_label.pack()
progress_bar = ttk.Progressbar(progress_frame, orient="horizontal", length=300, mode="determinate")
progress_bar.pack()

root.mainloop()
