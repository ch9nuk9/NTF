import os
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import threading
import tensorflow as tf
from datetime import datetime

# Define global constants and variables
output_directory = "output_directory"

# Function to read data from a .txt file
def read_data(file_path):
    try:
        data = pd.read_csv(file_path, delim_whitespace=True, header=None, names=['Frame', 'Time', 'X', 'Y'])
        return data
    except Exception as e:
        log_error(f"Error reading file {file_path}: {e}")
        return None

# Function to calculate parameters from tracking data
def calculate_parameters(data):
    data['Distance'] = data.apply(lambda row: euclidean((row['X'], row['Y']), (row['X'].shift(), row['Y'].shift())), axis=1)
    data['Distance'] = data['Distance'].fillna(0)
    data['TotalDistance'] = data['Distance'].cumsum()
    data['Displacement'] = euclidean((data['X'].iloc[0], data['Y'].iloc[0]), (data['X'].iloc[-1], data['Y'].iloc[-1]))
    data['Velocity_X'] = data['X'].diff() / data['Time'].diff()
    data['Velocity_Y'] = data['Y'].diff() / data['Time'].diff()
    data['Speed'] = np.sqrt(data['Velocity_X']**2 + data['Velocity_Y']**2)
    data['TurnAngle'] = np.arctan2(data['Y'].diff(), data['X'].diff()).diff()
    data['TurnAngle'] = data['TurnAngle'].fillna(0)
    return data

# Function to normalize parameters
def normalize_parameters(data):
    scaler = StandardScaler()
    data[['Distance', 'TotalDistance', 'Velocity_X', 'Velocity_Y', 'Speed', 'TurnAngle']] = scaler.fit_transform(
        data[['Distance', 'TotalDistance', 'Velocity_X', 'Velocity_Y', 'Speed', 'TurnAngle']])
    return data

# Function to compute summary statistics
def compute_statistics(data):
    statistics = {}
    parameters = ['Distance', 'TotalDistance', 'Velocity_X', 'Velocity_Y', 'Speed', 'TurnAngle']
    for param in parameters:
        statistics[param] = {
            'mean': data[param].mean(),
            'median': data[param].median(),
            'mode': data[param].mode()[0],
            'max': data[param].max(),
            'min': data[param].min(),
            'range': data[param].max() - data[param].min(),
            'std_dev': data[param].std(),
            'variance': data[param].var(),
            'skewness': data[param].skew(),
            'kurtosis': data[param].kurt()
        }
    return statistics

# Function to save data and statistics to CSV
def save_to_csv(data, statistics, output_path):
    data.to_csv(os.path.join(output_path, 'Estimates.csv'), index=False)
    stats_df = pd.DataFrame(statistics)
    stats_df.to_csv(os.path.join(output_path, 'SummaryStatistics.csv'))

# Function to perform clustering
def perform_clustering(data, method='kmeans', n_clusters=3):
    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters)
    elif method == 'hierarchical':
        model = AgglomerativeClustering(n_clusters=n_clusters)
    elif method == 'dbscan':
        model = DBSCAN()
    elif method == 'spectral':
        model = SpectralClustering(n_clusters=n_clusters)
    else:
        raise ValueError("Invalid clustering method")

    clusters = model.fit_predict(data)
    silhouette_avg = silhouette_score(data, clusters)
    return clusters, silhouette_avg

# Function to log errors
def log_error(message):
    with open('error_log.txt', 'a') as f:
        f.write(f"{datetime.now()} - {message}\n")

# GUI Implementation
class NematodeTrackingGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Nematode Tracking Analysis")
        self.geometry("600x400")
        
        self.main_directory = tk.StringVar()
        self.clustering_method = tk.StringVar(value='kmeans')
        self.n_clusters = tk.IntVar(value=3)
        
        self.create_widgets()
    
    def create_widgets(self):
        tk.Label(self, text="Main Directory:").pack(pady=10)
        tk.Entry(self, textvariable=self.main_directory, width=50).pack(pady=5)
        tk.Button(self, text="Browse", command=self.browse_directory).pack(pady=5)
        
        tk.Label(self, text="Clustering Method:").pack(pady=10)
        tk.OptionMenu(self, self.clustering_method, 'kmeans', 'hierarchical', 'dbscan', 'spectral').pack(pady=5)
        
        tk.Label(self, text="Number of Clusters:").pack(pady=10)
        tk.Entry(self, textvariable=self.n_clusters, width=5).pack(pady=5)
        
        self.progress_bar = ttk.Progressbar(self, orient="horizontal", mode="determinate", length=400)
        self.progress_bar.pack(pady=20)
        
        tk.Button(self, text="Start Analysis", command=self.start_analysis).pack(pady=20)
        
    def browse_directory(self):
        directory = filedialog.askdirectory()
        if directory:
            self.main_directory.set(directory)
    
    def start_analysis(self):
        threading.Thread(target=self.process_data).start()
    
    def process_data(self):
        main_directory = self.main_directory.get()
        if not main_directory:
            messagebox.showerror("Error", "Please select a main directory.")
            return
        
        output_path = os.path.join(main_directory, output_directory)
        os.makedirs(output_path, exist_ok=True)
        
        all_data = []
        subdirs = os.listdir(main_directory)
        self.progress_bar["maximum"] = len(subdirs)
        self.progress_bar["value"] = 0
        
        for subdir in subdirs:
            subdir_path = os.path.join(main_directory, subdir)
            if os.path.isdir(subdir_path):
                for file in os.listdir(subdir_path):
                    if file.endswith('.txt'):
                        file_path = os.path.join(subdir_path, file)
                        data = read_data(file_path)
                        if data is not None:
                            try:
                                data = calculate_parameters(data)
                                data = normalize_parameters(data)
                                statistics = compute_statistics(data)
                                save_to_csv(data, statistics, output_path)
                                all_data.append(data)
                            except Exception as e:
                                log_error(f"Error processing file {file_path}: {e}")
            self.progress_bar["value"] += 1
            self.update_idletasks()
        
        aggregated_data = pd.concat(all_data)
        aggregated_data.to_csv(os.path.join(output_path, 'Aggregated.csv'), index=False)
        
        clusters, silhouette_avg = perform_clustering(aggregated_data, method=self.clustering_method.get(), n_clusters=self.n_clusters.get())
        aggregated_data['Cluster'] = clusters
        aggregated_data.to_csv(os.path.join(output_path, 'ClusteredData.csv'), index=False)
        
        with open(os.path.join(output_path, 'ClusteringResults.txt'), 'w') as f:
            f.write(f"Silhouette Score: {silhouette_avg}\n")
        
        messagebox.showinfo("Analysis Complete", "Data analysis and clustering completed successfully.")

if __name__ == "__main__":
    app = NematodeTrackingGUI()
    app.mainloop()
