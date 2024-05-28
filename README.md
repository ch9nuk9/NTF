# NTF: Nematode Tracking Data Filter and Mock Dataset Creator Tool
## Overview
This repository contains the Nematode Tracking Dataset Filteration Tool (NTF), a comprehensive Python-based tool for analyzing and clustering nematode movement data. In addition, it includes a utility for creating mock test datasets, enabling users to generate synthetic data for testing and validation purposes.

## Key Features
- Data Processing: Reads raw data from text files, calculates movement parameters, and normalizes the data.
- Statistical Analysis: Computes basic statistical measures and generates histograms for each parameter.
- Clustering: Supports multiple clustering algorithms including KMeans, Hierarchical, DBSCAN, and Gaussian Mixture.
- Visualization: Provides 2D visualization of clustered data using PCA and Plotly.
- GUI Interface: Includes a user-friendly GUI for directory selection, clustering method configuration, and progress tracking.
- Error Logging: Logs errors and processing information to an error log file.
- Mock Data Generator: Generates synthetic nematode movement datasets for testing and validation.

## Prerequisites
- Python 3.x (Libraries: pandas, numpy, matplotlib, plotly, scikit-learn, tkinter, threading, logging, shutil)
- Anaconda or Miniconda
- Windows OS for .exe creation

## Installation and Setup 
## Windows (10 or 11, x86_64)
### Setting Up the Environment

1. **Create a New Conda Environment**:
    ```bash
    conda create -n NTF python=3.9
    conda activate NTF
    ```

2. **Install Dependencies**:
    ```bash
    conda install pandas numpy matplotlib plotly scikit-learn tkinter threading logging shutil
    ```

3. **Clone the Repository**:
    ```bash
    git clone https://github.com/ch9nuk9/NTF.git
    ```
    
4. **Change the Directory**:
    ```bash
    cd (path/to/repo)
    ```
    
5. **Install packages**:
    ```bash
    conda install -r requirements.txt
    ```
    
6. **Run the python scripts as needed**:
    ```bash
    python NTF_testbuild_v2.0_Final.py
    ```
    
6. **Use the GUI as needed**
     



### OR (Discouraged)

1. **Simply run the setup_environment.yaml**:
    ```bash
    conda env create -f setup_environment.yaml
    conda activate NTF

### OR (Discouraged)

1. **Simply run the setup_environment.sh (Creates the conda environment NTF, activates and installs all the dependencies within the NTF)**:
    ```bash
    bash setup_environment.sh



## Running the Script

1. **Activate the Environment**:
    ```bash
    conda activate NTF
    ```

2. **Change the Directory**:
    ```bash
    cd (path/to/repo)
    ```

3. **Run the Script**:
    ```bash
    python NTF_testbuild_v2.0_Final.py
    ```
    
    
## Creating a Standalone Executable (Only if necessary)

1. **Install PyInstaller**:
    ```bash
    conda install pyinstaller
    ```

2. **Change the Directory**:
    ```bash
    cd (path/to/repo)
    ```
    
3. **Generate the Executable**:
    ```bash
    pyinstaller --onefile NTF_testbuild_v2.0_Final.py
    ```

4. **Locate the Executable**:
   The executable will be located in the `dist` directory.

### Optional: Creating an Installer

To create an installer for the .exe file, you can use [Inno Setup](http://www.jrsoftware.org/isinfo.php). Follow the Inno Setup wizard to package your executable into an installer.



## Mock Data Generator (Only for test purposes)

1. **Navigate to the mock data generator directory**:
    ```bash
    cd (path/to/repo)
    ```

2. **Change the Directory**:
    ```bash
    python Data_generator_v1.0.py
    ```
       

## Usage
### Final Script

1. **Import Statements**: These import statements bring in necessary modules and libraries for data manipulation, plotting, clustering, and GUI creation.
    ```bash
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
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
    ```


2. **Logging Configuration**: Configures logging to capture error messages and log them to error.log.
    ```bash
    logging.basicConfig(filename='error.log', level=logging.ERROR, 
                    format='%(asctime)s:%(levelname)s:%(message)s')
    ```

    
3. **Data Processing Functions**: Calculates various parameters such as distance, displacement, velocities, and angles from the X, Y coordinates and Time.
   calculate_parameters - Calculates various parameters such as distance, displacement, velocities, and angles from the X, Y coordinates and Time.
   ```bash
    def calculate_parameters(data):
    # Calculate additional parameters based on X, Y coordinates and Time
    # Add columns for dx, dy, distance, total_distance, displacement, velocities, speed, and turn_angle
    return data
    ```

   normalize_data - Normalizes the data to have a mean of 0 and a standard deviation of 1.
   ```bash
    def normalize_data(data):
    # Normalize data using StandardScaler from scikit-learn
    return pd.DataFrame(normalized_data, columns=data.columns)
    ```

   compute_statistics - Computes summary statistical measures for the data.
   ```bash
    def compute_statistics(data):
    # Compute summary statistics such as mean, median, max, min, range, std_dev, variance, skewness, and kurtosis
    return pd.DataFrame(stats)
    ```

   generate_histograms - Creates and saves histograms for each column in the dataset.
   ```bash
    def generate_histograms(data, folder_path):
    # Generate histograms for each column in the data and save them as PNG files
    ```

   read_data_from_folder - Reads data from text files in a specified folder and returns a list of dataframes.
   ```bash
    def read_data_from_folder(folder_path):
    # Reads all .txt files from the specified folder and returns a list of dataframes
    return data_list
    ```

   
4. **Clustering Functions**:
   perform_clustering - Applies the specified clustering algorithm to the data.
   ```bash
    def perform_clustering(data, method, n_clusters=None, eps=None, min_samples=None):
    # Perform clustering using the specified method (KMeans, Hierarchical, DBSCAN, Gaussian Mixture)
    return labels, model
    ```

   plot_clusters - Reduces data to 2D using PCA and plots the clusters using Plotly.
   ```bash
    def plot_clusters(data, labels, output_dir):
    # Uses PCA to reduce data dimensions to 2D and then uses Plotly to plot clusters
    ```

   
5. **Directory Management**:
   create_output_directories - Creates directories for storing logs, processed data, and clustering results.
   ```bash
    def create_output_directories(base_dir):
    # Creates necessary output directories for storing logs, stationary and non-stationary data
    return output_dir, log_dir, stationary_dir, non_stationary_dir
    ```

   
6. **GUI Implementation**:
   NematodeTrackerApp - Implements the main application window with options to select a directory, etc.
   ```bash
    class NematodeTrackerApp:
    def __init__(self, root):
        # Initializes the GUI components
        self.create_widgets()

    def create_widgets(self):
        # Creates and arranges the widgets in the GUI
    ```

   select_directory - Allows users to select a directory containing data files.
   ```bash
    def select_directory(self):
    # Opens a file dialog to select a directory and logs the selected path
    ```

   show_clustering_info - Displays information on clustering method and adjusts input fields based on the method's requirements.
   ```bash
    def show_clustering_info(self, event):
    # Displays information about the selected clustering method and adjusts parameter input fields accordingly
    ```

   process_data - Starts the data processing in a new thread to keep the GUI responsive.
   ```bash
    def process_data(self):
    # Starts a new thread to process data to avoid freezing the GUI
    ```

   process_data_thread - Processes data, etc. This function runs in a separate thread to keep the GUI responsive.
   ```bash
    def process_data_thread(self):
    # Core data processing function executed in a separate thread
    try:
        # Reads and processes data from subfolders, calculates parameters, and performs clustering
    except Exception as e:
        logging.error("Error in processing data", exc_info=True)
        self.log.insert(tk.END, f"Error: {e}\n")
    ```


7. **Main Function**:
   The entry point of the script that initializes and runs the GUI application.
   ```bash
    def main():
    root = tk.Tk()
    app = NematodeTrackerApp(root)
    root.mainloop()
    ```


This script provides a full pipeline for processing and analyzing nematode movement data, from reading raw data files to generating statistical summaries and clustering results, all through an interactive GUI. It is designed to handle multiple datasets, normalize and compute necessary parameters, and visually present clustering results, making it a comprehensive tool for researchers working with such data.

For detailed information on each function and its parameters, refer to the comments in the code.


### Contributions
Contributions are welcome! Please open an issue to discuss your idea or submit a pull request with your changes.

### License:
This project is licensed under the MIT License.
