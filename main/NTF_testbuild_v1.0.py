import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MeanShift, SpectralClustering, Birch, OPTICS
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
        'kurtosis': data.kurt(),
        'outliers': ((data < (data.mean() - 3 * data.std())) | (data > (data.mean() + 3 * data.std()))).sum()
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

def read_data_from_folder(folder_path):
    data_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    logging.info(f"Found {len(data_files)} .txt files in {folder_path}")
    data_list = []
    for file in data_files:
        file_path = os.path.join(folder_path, file)
        try:
            data = pd.read_csv(file_path, sep='\s+', names=['Frame', 'Time', 'X', 'Y'])
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

def perform_clustering(data, method, n_clusters):
    if method == 'KMeans':
        model = KMeans(n_clusters=n_clusters)
    elif method == 'Hierarchical':
        model = AgglomerativeClustering(n_clusters=n_clusters)
    elif method == 'DBSCAN':
        model = DBSCAN()
    elif method == 'Gaussian Mixture':
        model = GaussianMixture(n_components=n_clusters)
    elif method == 'Mean Shift':
        model = MeanShift()
    elif method == 'Spectral':
        model = SpectralClustering(n_clusters=n_clusters)
    elif method == 'BIRCH':
        model = Birch(n_clusters=n_clusters)
    elif method == 'OPTICS':
        model = OPTICS()
    else:
        raise ValueError("Unknown clustering method")
    
    labels = model.fit_predict(data)
    return labels

def plot_clusters(data, labels, output_dir):
    pca = PCA(n_components=2)
    components = pca.fit_transform(data)
    fig = px.scatter(components, x=0, y=1, color=labels, title="Cluster Visualization")
    plot_path = os.path.join(output_dir, 'cluster_visualization.html')
    fig.write_html(plot_path)

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
            'KMeans', 'Hierarchical', 'DBSCAN', 'Gaussian Mixture', 'Mean Shift',
            'Spectral', 'BIRCH', 'OPTICS'
        ])
        self.cluster_method.set('DBSCAN')  # Set DBSCAN as the default value
        self.cluster_method.pack(pady=10)
        self.cluster_method.bind("<<ComboboxSelected>>", self.show_clustering_info)

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
            'DBSCAN': 'DBSCAN: Density-based, works with clusters of arbitrary shape, handles noise.',
            'Gaussian Mixture': 'Gaussian Mixture: Probabilistic clustering, suitable for overlapping clusters.',
            'Mean Shift': 'Mean Shift: Works well with clusters of different shapes, adaptive to cluster density.',
            'Spectral': 'Spectral Clustering: Suitable for clusters with complex shapes, works with numerical data.',
            'BIRCH': 'BIRCH: Suitable for large datasets, hierarchical clustering, works with numerical data.',
            'OPTICS': 'OPTICS: Density-based, handles clusters of varying densities, works with noise.'
        }
        messagebox.showinfo("Clustering Method Information", info.get(method, "No information available"))

    def process_data(self):
        self.log.insert(tk.END, "Processing data...\n")
        self.progress['value'] = 0
        threading.Thread(target=self.process_data_thread).start()

    def process_data_thread(self):
        try:
            output_dir = os.path.join(self.directory, "Output")
            os.makedirs(output_dir, exist_ok=True)
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
                    generate_histograms(data, subfolder)
                    estimates_file = os.path.join(subfolder, 'Estimates.csv')
                    data.to_csv(estimates_file, index=False)
                    aggregated_data.append(stats)

                processed_folders += 1
                self.progress['value'] = (processed_folders / total_folders) * 100
                self.root.update_idletasks()

            if aggregated_data:
                aggregated_df = pd.concat(aggregated_data)
                aggregated_file = os.path.join(output_dir, 'Aggregated.csv')
                aggregated_df.to_csv(aggregated_file, index=False)
                
                # Normalize aggregated data for clustering
                normalized_aggregated_df = normalize_data(aggregated_df)
                clustering_method = self.cluster_method.get()
                n_clusters = 3  # Default value, can be modified based on user input
                labels = perform_clustering(normalized_aggregated_df, clustering_method, n_clusters)

                # Store clustering results
                aggregated_df['Cluster'] = labels
                results_file = os.path.join(output_dir, 'Results.csv')
                aggregated_df.to_csv(results_file, index=False)
                
                # Plot clusters
                plot_clusters(normalized_aggregated_df, labels, output_dir)
                
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
