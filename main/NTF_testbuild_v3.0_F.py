import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-GUI plotting
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import logging
import shutil

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
        'max': data.max(),
        'min': data.min(),
        'range': data.max() - data.min(),
        'std_dev': data.std(),
        'variance': data.var(),
        'skewness': data.skew(),
        'kurtosis': data.kurt(),
    }
    return pd.DataFrame(stats)

def generate_histograms(data, folder_path):
    for column in data.columns:
        plt.figure()
        data[column].hist(bins=50)
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(folder_path, f'{column}_histogram.png'))
        plt.close()

def read_data_from_folder(folder_path, fps=None):
    data_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    logging.info(f"Found {len(data_files)} .txt files in {folder_path}")
    data_list = []
    for file in data_files:
        file_path = os.path.join(folder_path, file)
        try:
            data = pd.read_csv(file_path, sep=r'\s*,\s*', names=['Frame', 'Time', 'X', 'Y'], engine='python')  # Use comma as separator
            data['Frame'] = pd.to_numeric(data['Frame'], errors='coerce')
            data['Time'] = pd.to_numeric(data['Time'], errors='coerce')
            data['X'] = pd.to_numeric(data['X'], errors='coerce')
            data['Y'] = pd.to_numeric(data['Y'], errors='coerce')
            data.dropna(subset=['Frame', 'X', 'Y'], inplace=True)  # Drop rows with NaN values in essential columns

            if data['Time'].isnull().all() and fps is not None:
                # Compute time values if FPS is provided and Time column is missing
                data['Time'] = data['Frame'] / fps
                new_file_path = file_path.replace('.txt', '_with_time.txt')
                data.to_csv(new_file_path, sep=',', index=False)
                file_path = new_file_path
            
            data_list.append(data)
            logging.info(f"Successfully read {file_path}")
        except Exception as e:
            logging.error(f"Error reading {file_path}: {e}")
    return data_list

def perform_clustering(data, method, n_clusters=None, eps=None, min_samples=None):
    if method == 'KMeans':
        model = KMeans(n_clusters=n_clusters)
    elif method == 'Hierarchical':
        model = AgglomerativeClustering(n_clusters=n_clusters)
    elif method == 'DBSCAN':
        model = DBSCAN(eps=eps, min_samples=min_samples)
    elif method == 'Gaussian Mixture':
        model = GaussianMixture(n_components=n_clusters)
    else:
        raise ValueError("Unknown clustering method")
    
    labels = model.fit_predict(data)
    return labels, model

def plot_clusters(data, labels, output_dir):
    pca = PCA(n_components=2)
    components = pca.fit_transform(data)
    fig = px.scatter(components, x=0, y=1, color=labels, title="Cluster Visualization")
    plot_path = os.path.join(output_dir, 'cluster_visualization.html')
    fig.write_html(plot_path)

def create_output_directories(base_dir):
    output_dir = os.path.join(base_dir, "Output")
    log_dir = os.path.join(output_dir, "logs")
    stationary_dir = os.path.join(output_dir, "stationary")
    non_stationary_dir = os.path.join(output_dir, "non_stationary")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(stationary_dir, exist_ok=True)
    os.makedirs(non_stationary_dir, exist_ok=True)
    return output_dir, log_dir, stationary_dir, non_stationary_dir

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

        self.cluster_label = ttk.Label(self.root, text="Select Clustering Method:")
        self.cluster_label.pack(pady=10)

        self.cluster_method = ttk.Combobox(self.root, values=[
            'KMeans', 'Hierarchical', 'DBSCAN', 'Gaussian Mixture'
        ])
        self.cluster_method.set('DBSCAN')  # Set DBSCAN as the default value
        self.cluster_method.pack(pady=10)
        self.cluster_method.bind("<<ComboboxSelected>>", self.show_clustering_info)

        self.param_frame = ttk.Frame(self.root)
        self.param_frame.pack(pady=10)

        self.n_clusters_label = ttk.Label(self.param_frame, text="Number of Clusters:")
        self.n_clusters_label.grid(row=0, column=0, padx=5)
        self.n_clusters_entry = ttk.Entry(self.param_frame)
        self.n_clusters_entry.grid(row=0, column=1, padx=5)
        
        self.eps_label = ttk.Label(self.param_frame, text="EPS:")
        self.eps_label.grid(row=1, column=0, padx=5)
        self.eps_entry = ttk.Entry(self.param_frame)
        self.eps_entry.grid(row=1, column=1, padx=5)
        
        self.min_samples_label = ttk.Label(self.param_frame, text="Min Samples:")
        self.min_samples_label.grid(row=2, column=0, padx=5)
        self.min_samples_entry = ttk.Entry(self.param_frame)
        self.min_samples_entry.grid(row=2, column=1, padx=5)

        self.fps_label = ttk.Label(self.root, text="Frames Per Second (FPS):")
        self.fps_label.pack(pady=10)
        
        self.fps_entry = ttk.Entry(self.root)
        self.fps_entry.pack(pady=10)

        self.progress = ttk.Progressbar(self.root, orient='horizontal', mode='determinate', length=600)
        self.progress.pack(pady=20)

        self.log = tk.Text(self.root, height=15, width=80)
        self.log.pack(pady=10)

    def select_directory(self):
        self.directory = filedialog.askdirectory()
        self.log.insert(tk.END, f"Selected directory: {self.directory}\n")
        logging.info(f"Selected directory: {self.directory}")

    def show_clustering_info(self, event):
        method = self.cluster_method.get()
        info = {
            'KMeans': 'KMeans Clustering: Suitable for well-separated clusters, works with numerical data.',
            'Hierarchical': 'Hierarchical Clustering: Suitable for nested clusters, works with numerical data.',
            'DBSCAN': 'DBSCAN: Density-based, works with clusters of arbitrary shape, handles noise. Requires EPS and Min Samples.',
            'Gaussian Mixture': 'Gaussian Mixture: Probabilistic clustering, suitable for overlapping clusters.',
        }
        messagebox.showinfo("Clustering Method Information", info.get(method, "No information available"))
        
        # Show or hide parameter fields based on the selected method
        if method in ['KMeans', 'Hierarchical', 'Gaussian Mixture']:
            self.n_clusters_label.grid()
            self.n_clusters_entry.grid()
            self.eps_label.grid_remove()
            self.eps_entry.grid_remove()
            self.min_samples_label.grid_remove()
            self.min_samples_entry.grid_remove()
        elif method == 'DBSCAN':
            self.n_clusters_label.grid_remove()
            self.n_clusters_entry.grid_remove()
            self.eps_label.grid()
            self.eps_entry.grid()
            self.min_samples_label.grid()
            self.min_samples_entry.grid()
        else:
            self.n_clusters_label.grid_remove()
            self.n_clusters_entry.grid_remove()
            self.eps_label.grid_remove()
            self.eps_entry.grid_remove()
            self.min_samples_label.grid_remove()
            self.min_samples_entry.grid_remove()

    def process_data(self):
        self.log.insert(tk.END, "Processing data...\n")
        self.progress['value'] = 0
        threading.Thread(target=self.process_data_thread).start()

    def process_data_thread(self):
        try:
            subfolders = [f.path for f in os.scandir(self.directory) if f.is_dir()]
            output_dir, log_dir, stationary_dir, non_stationary_dir = create_output_directories(self.directory)
            
            total_folders = len(subfolders)
            processed_folders = 0

            aggregated_data = []

            fps_value = float(self.fps_entry.get()) if self.fps_entry.get() else None

            for subfolder in subfolders:
                relative_path = os.path.relpath(subfolder, self.directory)
                output_subfolder = os.path.join(output_dir, relative_path)
                os.makedirs(output_subfolder, exist_ok=True)
                
                self.log.insert(tk.END, f"Processing subfolder: {subfolder}\n")
                logging.info(f"Processing subfolder: {subfolder}")
                
                data_list = read_data_from_folder(subfolder, fps=fps_value)
                if not data_list:
                    self.log.insert(tk.END, f"No .txt files found in {subfolder}\n")
                    logging.info(f"No .txt files found in {subfolder}")
                    continue
                
                for data in data_list:
                    data = calculate_parameters(data)
                    normalized_data = normalize_data(data)
                    stats = compute_statistics(normalized_data)
                    generate_histograms(data, output_subfolder)
                    
                    estimates_file = os.path.join(output_subfolder, 'Estimates.csv')
                    data.to_csv(estimates_file, index=False)
                    aggregated_data.append(stats)
                    
                    # Check if the data is stationary
                    if (data['velocity_x'].std() < 1e-2) and (data['velocity_y'].std() < 1e-2):
                        dest_subfolder = os.path.join(stationary_dir, relative_path)
                    else:
                        dest_subfolder = os.path.join(non_stationary_dir, relative_path)
                    
                    data_file_name = os.path.basename(estimates_file)
                    os.makedirs(dest_subfolder, exist_ok=True)
                    data.to_csv(os.path.join(dest_subfolder, data_file_name), index=False)

                    # Copy the original dataset file to the appropriate folder
                    for file in os.listdir(subfolder):
                        if file.endswith('.txt'):
                            shutil.copy2(os.path.join(subfolder, file), dest_subfolder)

                processed_folders += 1
                self.progress['value'] = (processed_folders / total_folders) * 100
                self.root.update_idletasks()

            if aggregated_data:
                aggregated_df = pd.concat(aggregated_data)
                aggregated_file = os.path.join(output_dir, 'Aggregated.csv')
                aggregated_df.to_csv(aggregated_file, index=False)
                
                # Select relevant features for clustering
                relevant_features = [
                    'mean', 'median', 'max', 'min', 
                    'range', 'std_dev', 'variance', 'skewness', 'kurtosis'
                ]
                aggregated_df = aggregated_df[relevant_features]

                # Normalize aggregated data for clustering
                normalized_aggregated_df = normalize_data(aggregated_df)

                # Apply PCA for dimensionality reduction
                pca = PCA(n_components=2)
                pca_data = pca.fit_transform(normalized_aggregated_df)

                clustering_method = self.cluster_method.get()
                n_clusters = int(self.n_clusters_entry.get()) if self.n_clusters_entry.get() else 3
                eps = float(self.eps_entry.get()) if self.eps_entry.get() else 0.5
                min_samples = int(self.min_samples_entry.get()) if self.min_samples_entry.get() else 5
                
                if clustering_method == 'DBSCAN':
                    labels, model = perform_clustering(pca_data, clustering_method, eps=eps, min_samples=min_samples)
                else:
                    labels, model = perform_clustering(pca_data, clustering_method, n_clusters=n_clusters)

                # Compute silhouette score to evaluate clustering
                silhouette_avg = silhouette_score(pca_data, labels)
                self.log.insert(tk.END, f"Silhouette Score: {silhouette_avg}\n")
                logging.info(f"Silhouette Score: {silhouette_avg}")

                # Store clustering results
                aggregated_df['Cluster'] = labels
                results_file = os.path.join(output_dir, 'Results.csv')
                aggregated_df.to_csv(results_file, index=False)
                
                # Plot clusters
                plot_clusters(pca_data, labels, output_dir)
                
                self.log.insert(tk.END, "Data processing and clustering completed.\n")
                logging.info("Data processing and clustering completed.")
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
