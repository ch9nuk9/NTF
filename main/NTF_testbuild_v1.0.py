import os  # To interact with the operating system, e.g., for directory traversal
import numpy as np  # For numerical operations on arrays
import pandas as pd  # For data manipulation and analysis
import tkinter as tk  # For creating the graphical user interface (GUI)
from tkinter import filedialog, messagebox, ttk  # For file dialogs, message boxes, and themed widgets in Tkinter
from sklearn.preprocessing import StandardScaler  # For normalizing data
from sklearn.decomposition import PCA  # For principal component analysis
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering  # For clustering algorithms
from sklearn.metrics import silhouette_score  # For evaluating clustering performance
import matplotlib.pyplot as plt  # For plotting data
from scipy.spatial.distance import euclidean  # For calculating Euclidean distance between points
import threading  # For running tasks in separate threads
import tensorflow as tf  # For GPU acceleration and tensor computations
from datetime import datetime  # For timestamping logs

# Define global constants and variables
output_directory = "output_directory"
stationary_directory = "output_directory/Stationary"
nonstationary_directory = "output_directory/Nonstationary"
DISTANCE_THRESHOLD = 5  # Threshold for total distance to classify as artifact

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
            'mode': data[param].mode()[0] if not data[param].mode().empty else np.nan,
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

# Function to check if the data represents a stationary artifact
def is_stationary(data):
    total_distance = data['TotalDistance'].iloc[-1]
    x_variance = data['X'].var()
    y_variance = data['Y'].var()
    avg_speed = data['Speed'].mean()
    
    if total_distance < DISTANCE_THRESHOLD and x_variance < 1e-5 and y_variance < 1e-5 and avg_speed < 1e-3:
        return True
    return False

# Function to save a copy of the dataset to the respective folder
def save_dataset_copy(data, subdir, filename, output_folder):
    output_subdir = os.path.join(output_folder, subdir)
    os.makedirs(output_subdir, exist_ok=True)
    output_file_path = os.path.join(output_subdir, filename)
    data.to_csv(output_file_path, index=False)

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
        
        os.makedirs(stationary_directory, exist_ok=True)
        os.makedirs(nonstationary_directory, exist_ok=True)
        
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
                                if is_stationary(data):
                                    log_error(f"Stationary artifact detected in file {file_path}")
                                    save_dataset_copy(data, subdir, file, stationary_directory)
                                    continue
                                else:
                                    save_dataset_copy(data, subdir, file, nonstationary_directory)
                                data = normalize_parameters(data)
                                statistics = compute_statistics(data)
                                save_to_csv(data, statistics, nonstationary_directory)
                                all_data.append(data)
                            except Exception as e:
                                log_error(f"Error processing file {file_path}: {e}")
            self.progress_bar["value"] += 1
            self.update_idletasks()
        
        if all_data:
            aggregated_data = pd.concat(all_data)
            aggregated_data.to_csv(os.path.join(nonstationary_directory, 'Aggregated.csv'), index=False)
            
            clusters, silhouette_avg = perform_clustering(aggregated_data, method=self.clustering_method.get(), n_clusters=self.n_clusters.get())
            aggregated_data['Cluster'] = clusters
            aggregated_data.to_csv(os.path.join(nonstationary_directory, 'ClusteredData.csv'), index=False)
            
            with open(os.path.join(nonstationary_directory, 'ClusteringResults.txt'), 'w') as f:
                f.write(f"Silhouette Score: {silhouette_avg}\n")
            
            messagebox.showinfo("Analysis Complete", "Data analysis and clustering completed successfully.")
        else:
            messagebox.showinfo("No Valid Data", "No valid data found for analysis. All files were stationary artifacts.")

if __name__ == "__main__":
    app = NematodeTrackingGUI()
    app.mainloop()
