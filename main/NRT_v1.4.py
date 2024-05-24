import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA

def load_folder():
    folder_path = filedialog.askdirectory()
    if folder_path:
        process_folder(folder_path)

def process_folder(folder_path):
    all_filtered_data = []
    subfolders = [subfolder for subfolder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, subfolder))]
    total_subfolders = len(subfolders)
    valid_subfolder_count = 0
    skipped_subfolder_count = 0
    skipped_folders = []
    stationary_frames_record = []

    for idx, subfolder in enumerate(subfolders):
        subfolder_path = os.path.join(folder_path, subfolder)
        txt_files = [f for f in os.listdir(subfolder_path) if f.endswith('.txt')]
        if txt_files:
            txt_file_path = os.path.join(subfolder_path, txt_files[0])
            print(f"Processing file: {txt_file_path}")
            try:
                data = pd.read_csv(txt_file_path, sep=',')
            except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
                print(f"Error reading file {txt_file_path}: {e}")
                skipped_subfolder_count += 1
                skipped_folders.append(subfolder)
                update_progress(idx, total_subfolders)
                continue

            data['subject_id'] = subfolder
            data['dX'] = data['X'].diff().fillna(0)
            data['dY'] = data['Y'].diff().fillna(0)
            data['distance'] = np.sqrt(data['dX']**2 + data['dY']**2)

            if data['distance'].std() == 0:
                print(f"Subject {subfolder} has no movement variability. Skipping.")
                skipped_subfolder_count += 1
                skipped_folders.append(subfolder)
                update_progress(idx, total_subfolders)
                continue

            try:
                optimal_threshold = find_optimal_threshold(data)
            except Exception as e:
                print(f"Error finding optimal threshold for subject {subfolder}: {e}")
                skipped_subfolder_count += 1
                skipped_folders.append(subfolder)
                update_progress(idx, total_subfolders)
                continue

            print(f"Subject {subfolder} - Optimal Threshold: {optimal_threshold}")
            stationary = data['distance'] < optimal_threshold
            stationary_ratio = stationary.mean()
            print(f"Subject {subfolder} - Stationary Ratio: {stationary_ratio:.2f}")
            
            stationary_frames_record.append(stationary_ratio)

            if stationary_ratio > 0.95:
                print(f"Subject {subfolder} is predominantly stationary. Skipping.")
                skipped_subfolder_count += 1
                skipped_folders.append(subfolder)
                update_progress(idx, total_subfolders)
                continue

            filtered_data = data[~stationary]
            all_filtered_data.append(filtered_data)
            valid_subfolder_count += 1
        
        update_progress(idx, total_subfolders)

    if all_filtered_data:
        combined_filtered_data = pd.concat(all_filtered_data)
        save_path = filedialog.asksaveasfilename(defaultextension=".txt")
        if save_path:
            try:
                combined_filtered_data[['subject_id', 'frame', 'time', 'X', 'Y']].to_csv(save_path, sep=',', index=False)
                messagebox.showinfo("Success", "File has been saved successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save the file: {e}")
    else:
        messagebox.showwarning("No Data", "No valid data to save.")
    
    progress_bar['value'] = 100
    progress_label['text'] = 'Processing Complete'
    valid_label['text'] = f'Valid Subfolders: {valid_subfolder_count}'
    skipped_label['text'] = f'Skipped Subfolders: {skipped_subfolder_count}'
    
    visualize_stationary_vs_moving(stationary_frames_record)

def update_progress(current, total):
    progress = (current + 1) / total * 100
    progress_bar['value'] = progress
    progress_label['text'] = f'Processing {current + 1} of {total} subfolders'
    root.update_idletasks()

def find_optimal_threshold(data, manual_threshold=None):
    distances = data['distance'].values
    try:
        density = gaussian_kde(distances)
    except np.linalg.LinAlgError:
        pca = PCA(n_components=1)
        distances = pca.fit_transform(distances.reshape(-1, 1)).flatten()
        density = gaussian_kde(distances)
    xs = np.linspace(0, np.max(distances), 1000)
    density_values = density(xs)
    if manual_threshold is None:
        threshold = xs[np.argmax(density_values > density_values.max() * 0.05)]
    else:
        threshold = np.percentile(distances, manual_threshold)
    plt.figure(figsize=(10, 6))
    plt.plot(xs, density_values, label='Density')
    plt.axvline(threshold, color='r', linestyle='--', label=f'Optimal Threshold: {threshold:.2f}')
    plt.xlabel('Distance')
    plt.ylabel('Density')
    plt.title('Optimal Threshold Determination')
    plt.legend()
    plt.show()
    return threshold

def set_manual_threshold():
    try:
        manual_threshold = float(manual_threshold_entry.get())
        if 0 <= manual_threshold <= 100:
            find_optimal_threshold(data=None, manual_threshold=manual_threshold)
        else:
            messagebox.showerror("Error", "Threshold percentage must be between 0 and 100.")
    except ValueError:
        messagebox.showerror("Error", "Invalid input. Please enter a numerical value.")

def visualize_stationary_vs_moving(stationary_ratios):
    plt.figure(figsize=(10, 6))
    plt.hist(stationary_ratios, bins=20, alpha=0.7, label='Stationary Ratios')
    plt.axvline(0.95, color='r', linestyle='--', label='Threshold for Skipping')
    plt.xlabel('Stationary Ratio')
    plt.ylabel('Count')
    plt.title('Distribution of Stationary vs. Moving Elements')
    plt.legend()
    plt.show()

root = tk.Tk()
root.title("Nematode Tracking Filter")

load_button = tk.Button(root, text="Load Folder", command=load_folder)
load_button.pack(pady=10)

manual_threshold_label = tk.Label(root, text="Manual Threshold Percentage:")
manual_threshold_label.pack()
manual_threshold_entry = ttk.Entry(root)
manual_threshold_entry.pack()
manual_threshold_entry.insert(0, "5")
manual_threshold_button = tk.Button(root, text="Set Manual Threshold", command=set_manual_threshold)
manual_threshold_button.pack()

progress_frame = tk.Frame(root)
progress_frame.pack(pady=10)
progress_label = tk.Label(progress_frame, text="")
progress_label.pack()
progress_bar = ttk.Progressbar(progress_frame, orient="horizontal", length=250, mode="determinate")
progress_bar.pack()

count_frame = tk.Frame(root)
count_frame.pack(pady=10)
valid_label = tk.Label(count_frame, text="Valid Subfolders: 0")
valid_label.pack()
skipped_label = tk.Label(count_frame, text="Skipped Subfolders: 0")
skipped_label.pack()

root.mainloop()
