import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Progressbar
import json
from threading import Thread
import time

class ThresholdRefinementApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Threshold Refinement")
        
        # UI Elements
        self.load_button = tk.Button(root, text="Load Directory", command=self.load_directory)
        self.load_button.pack(pady=20)
        
        self.progress_label = tk.Label(root, text="Progress: 0/0")
        self.progress_label.pack()
        
        self.progress_bar = Progressbar(root, orient=tk.HORIZONTAL, length=300, mode='determinate')
        self.progress_bar.pack(pady=20)

        # Pause/Resume/Cancel Buttons
        self.pause_button = tk.Button(root, text="Pause", command=self.pause_processing, state=tk.DISABLED)
        self.pause_button.pack(side=tk.LEFT, padx=10)
        
        self.resume_button = tk.Button(root, text="Resume", command=self.resume_processing, state=tk.DISABLED)
        self.resume_button.pack(side=tk.LEFT, padx=10)
        
        self.cancel_button = tk.Button(root, text="Cancel", command=self.cancel_processing, state=tk.DISABLED)
        self.cancel_button.pack(side=tk.LEFT, padx=10)
        
        # Configuration
        self.config_file = 'config.json'
        self.load_config()

        # Variables
        self.folder_count = 0
        self.folders_processed = 0
        self.is_paused = False
        self.is_canceled = False
        self.thread = None

    def load_directory(self):
        dir_path = filedialog.askdirectory()
        if dir_path:
            self.folder_count = sum([len(files) for r, d, files in os.walk(dir_path) if any(file.endswith(".txt") for file in files)])
            self.progress_bar['maximum'] = self.folder_count
            self.process_directory(dir_path)
    
    def process_directory(self, dir_path):
        self.is_paused = False
        self.is_canceled = False
        self.thread = Thread(target=self._process_directory, args=(dir_path,))
        self.thread.start()
        self.pause_button.config(state=tk.NORMAL)
        self.cancel_button.config(state=tk.NORMAL)
    
    def _process_directory(self, dir_path):
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.endswith(".txt"):
                    while self.is_paused:
                        time.sleep(1)
                    if self.is_canceled:
                        return
                    file_path = os.path.join(root, file)
                    try:
                        self.process_file(file_path)
                    except Exception as e:
                        self.log_error(file_path, str(e))
                        messagebox.showerror("Error", f"Error processing file {file_path}: {str(e)}")
                    self.folders_processed += 1
                    self.progress_label.config(text=f"Progress: {self.folders_processed}/{self.folder_count}")
                    self.progress_bar['value'] = self.folders_processed
                    self.root.update_idletasks()
        messagebox.showinfo("Info", "Processing complete!")
        self.pause_button.config(state=tk.DISABLED)
        self.resume_button.config(state=tk.DISABLED)
        self.cancel_button.config(state=tk.DISABLED)
    
    def process_file(self, file_path):
        try:
            data = pd.read_csv(file_path, sep=r'\s+', names=['frame', 'time', 'X', 'Y'])
        except Exception as e:
            raise ValueError(f"Error reading file {file_path}: {e}")
        
        if data.empty:
            raise ValueError("File is empty or corrupted")
        
        data['dX'] = data['X'].diff().fillna(0)
        data['dY'] = data['Y'].diff().fillna(0)
        data['distance'] = np.sqrt(data['dX']**2 + data['dY']**2)
        
        optimal_threshold = self.find_optimal_threshold(data, file_path)
        print(f"Optimal Threshold for {file_path}: {optimal_threshold}")
        self.save_results(file_path, optimal_threshold)
    
    def find_optimal_threshold(self, data, file_path):
        distances = data['distance'].values
        distances = distances[distances > 0]  # Exclude zero distances
        
        if len(distances) < 2:
            raise ValueError("Not enough data points for KDE or PCA.")
        
        try:
            density = gaussian_kde(distances)
            xs = np.linspace(0, np.max(distances), 1000)
            density_values = density(xs)
            threshold = xs[np.argmax(density_values > density_values.max() * self.threshold_percentage)]
        except np.linalg.LinAlgError:
            pca = PCA(n_components=1)
            distances_reshaped = distances.reshape(-1, 1)
            pca.fit(distances_reshaped)
            xs = np.linspace(0, np.max(distances), 1000)
            fp = pca.transform(distances_reshaped).flatten()
            if len(fp) != len(xs):
                fp = np.interp(xs, np.linspace(0, np.max(fp), len(fp)), fp)
            density_values = fp
            threshold = xs[np.argmax(density_values > density_values.max() * self.threshold_percentage)]
        
        # Plotting the density and threshold
        self.plot_density(xs, density_values, threshold, file_path)
        
        return threshold
    
    def plot_density(self, xs, density_values, threshold, file_path):
        plt.figure(figsize=(10, 6))
        plt.plot(xs, density_values, label='Density')
        plt.axvline(threshold, color='r', linestyle='--', label=f'Optimal Threshold: {threshold:.2f}')
        plt.xlabel('Distance')
        plt.ylabel('Density')
        plt.title('Optimal Threshold Determination')
        plt.legend()
        
        # Save the plot without displaying it
        plot_file = file_path.replace('.txt', '_threshold_plot.png')
        plt.savefig(plot_file)
        plt.close()
    
    def save_results(self, file_path, threshold):
        result_file = file_path.replace('.txt', '_threshold.txt')
        with open(result_file, 'w') as f:
            f.write(f"Optimal Threshold: {threshold}\n")
    
    def pause_processing(self):
        self.is_paused = True
        self.pause_button.config(state=tk.DISABLED)
        self.resume_button.config(state=tk.NORMAL)
    
    def resume_processing(self):
        self.is_paused = False
        self.pause_button.config(state=tk.NORMAL)
        self.resume_button.config(state=tk.DISABLED)
    
    def cancel_processing(self):
        self.is_canceled = True
        self.pause_button.config(state=tk.DISABLED)
        self.resume_button.config(state=tk.DISABLED)
        self.cancel_button.config(state=tk.DISABLED)
    
    def log_error(self, file_path, error_message):
        log_file = "error_log.txt"
        with open(log_file, "a") as log:
            log.write(f"Error processing file {file_path}: {error_message}\n")
    
    def load_config(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                self.threshold_percentage = config.get('threshold_percentage', 0.05)
        else:
            self.threshold_percentage = 0.05
    
    def save_config(self):
        config = {
            'threshold_percentage': self.threshold_percentage,
        }
        with open(self.config_file, 'w') as f:
            json.dump(config, f)

# Create the main window
root = tk.Tk()
app = ThresholdRefinementApp(root)

# Run the application
root.mainloop()
