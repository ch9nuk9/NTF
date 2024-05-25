import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import logging

logging.basicConfig(filename='error.log', level=logging.ERROR, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

def calculate_parameters(data):
    data['dx'] = data['X'].diff().fillna(0)
    data['dy'] = data['Y'].diff().fillna(0)
    data['distance'] = np.sqrt(data['dx']**2 + data['dy']**2)
    data['total_distance'] = data['distance'].cumsum()
    data['displacement'] = np.sqrt((data['X'] - data['X'].iloc[0])**2 + (data['Y'] - data['Y'].iloc[0])**2)
    data['velocity_x'] = data['dx'] / data['Time'].diff().fillna(1)
    data['velocity_y'] = data['dy'] / data['Time'].diff().fillna(1)
    data['speed'] = data['distance'] / data['Time'].diff().fillna(1)
    data['turn_angle'] = np.arctan2(data['dy'], data['dx']).diff().fillna(0)
    return data

def normalize_data(data):
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)
    return pd.DataFrame(normalized_data, columns=data.columns)

def compute_statistics(data):
    stats = {
        'mean': data.mean(),
        'median': data.median(),
        'mode': data.mode().iloc[0],
        'max': data.max(),
        'min': data.min(),
        'range': data.max() - data.min(),
        'std_dev': data.std(),
        'std_err': data.sem(),
        'variance': data.var(),
        'skewness': data.skew(),
        'kurtosis': data.kurt()
    }
    return pd.DataFrame(stats)

def read_data_from_folder(folder_path):
    data_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    logging.info(f"Found {len(data_files)} .txt files in {folder_path}")
    data_list = []
    for file in data_files:
        file_path = os.path.join(folder_path, file)
        try:
            data = pd.read_csv(file_path, sep='\s+', names=['Frame', 'Time', 'X', 'Y'])
            # Convert columns to numeric, coerce errors to NaN
            data['Frame'] = pd.to_numeric(data['Frame'], errors='coerce')
            data['Time'] = pd.to_numeric(data['Time'], errors='coerce')
            data['X'] = pd.to_numeric(data['X'], errors='coerce')
            data['Y'] = pd.to_numeric(data['Y'], errors='coerce')
            data.dropna(inplace=True)  # Drop rows with NaN values
            data_list.append(data)
            logging.info(f"Successfully read {file_path}")
        except Exception as e:
            logging.error(f"Error reading {file_path}: {e}")
    return data_list

class NematodeTrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Nematode Tracker Analysis")
        self.root.geometry("800x600")

        self.create_widgets()

    def create_widgets(self):
        self.label = ttk.Label(self.root, text="Select Data Directory:")
        self.label.pack(pady=10)

        self.dir_button = ttk.Button(self.root, text="Browse", command=self.select_directory)
        self.dir_button.pack(pady=10)

        self.process_button = ttk.Button(self.root, text="Process Data", command=self.process_data)
        self.process_button.pack(pady=20)

        self.progress = ttk.Progressbar(self.root, orient='horizontal', mode='determinate', length=600)
        self.progress.pack(pady=20)

        self.log = tk.Text(self.root, height=15, width=80)
        self.log.pack(pady=10)

    def select_directory(self):
        self.directory = filedialog.askdirectory()
        self.log.insert(tk.END, f"Selected directory: {self.directory}\n")
        logging.info(f"Selected directory: {self.directory}")

    def process_data(self):
        self.log.insert(tk.END, "Processing data...\n")
        self.progress['value'] = 0
        threading.Thread(target=self.process_data_thread).start()

    def process_data_thread(self):
        try:
            subfolders = [f.path for f in os.scandir(self.directory) if f.is_dir()]
            total_folders = len(subfolders)
            processed_folders = 0

            aggregated_data = []

            for subfolder in subfolders:
                self.log.insert(tk.END, f"Processing subfolder: {subfolder}\n")
                logging.info(f"Processing subfolder: {subfolder}")
                data_list = read_data_from_folder(subfolder)
                if not data_list:
                    self.log.insert(tk.END, f"No .txt files found in {subfolder}\n")
                    logging.info(f"No .txt files found in {subfolder}")
                    continue
                for data in data_list:
                    data = calculate_parameters(data)
                    normalized_data = normalize_data(data)
                    stats = compute_statistics(normalized_data)
                    estimates_file = os.path.join(subfolder, 'Estimates.csv')
                    data.to_csv(estimates_file, index=False)
                    aggregated_data.append(stats)

                processed_folders += 1
                self.progress['value'] = (processed_folders / total_folders) * 100
                self.root.update_idletasks()

            if aggregated_data:
                aggregated_df = pd.concat(aggregated_data)
                aggregated_file = os.path.join(self.directory, 'Aggregated.csv')
                aggregated_df.to_csv(aggregated_file, index=False)
                results_file = os.path.join(self.directory, 'Results.csv')
                aggregated_df.to_csv(results_file, index=False)
                self.log.insert(tk.END, "Data processing completed.\n")
                logging.info("Data processing completed.")
            else:
                self.log.insert(tk.END, "No data processed. Check if the directory contains valid .txt files.\n")
                logging.info("No data processed. Check if the directory contains valid .txt files.")
        except Exception as e:
            logging.error("Error in processing data", exc_info=True)
            self.log.insert(tk.END, f"Error: {e}\n")

def main():
    root = tk.Tk()
    app = NematodeTrackerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
