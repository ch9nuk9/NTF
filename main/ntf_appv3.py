import sys
import os
import traceback
import numpy as np
import pandas as pd
import shutil
import json
import logging
from datetime import datetime
from PyQt5 import QtWidgets, QtCore, QtGui, QtWebEngineWidgets
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox, QLabel, QProgressBar,
    QVBoxLayout, QWidget, QPushButton, QLineEdit, QCheckBox, QComboBox,
    QHBoxLayout, QTextEdit, QGroupBox, QTabWidget, QSpinBox, QListWidget, QSplitter,
    QDoubleSpinBox
)
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QThread, QUrl
import plotly.express as px
import plotly.offline as po
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from concurrent.futures import ThreadPoolExecutor
import cv2  # OpenCV for video and image processing
from PIL import Image  # <-- Newly added import for handling .tif files
from PIL import Image
from threading import Lock

# Print the path where the .ntf_history.json file should be saved
print(os.path.join(os.path.expanduser('~'), '.ntf_history.json'))

# Global exception handler to catch uncaught exceptions
def handle_global_exception(exc_type, exc_value, exc_traceback):
    """
    Global exception handler for uncaught exceptions.
    Logs and shows the full traceback of the error.
    """
    error_message = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    logging.error("Uncaught Exception:", exc_info=(exc_type, exc_value, exc_traceback))
    
    # Display a detailed message box
    QMessageBox.critical(None, "Unhandled Exception", f"An unhandled exception occurred:\n\n{error_message}")

# Set the custom global exception handler
sys.excepthook = handle_global_exception

# Add the enhanced helper function for batch and single session validation
def validate_input_directory(input_dir, is_batch_mode):
    """
    Validates the input directory for either batch or single session processing.
    Batch mode expects session directories containing _track_ folders.
    """
    if is_batch_mode:
        # Top-level batch processing: Check if it contains valid session folders
        session_folders = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
        if not session_folders:
            raise ValueError("No session folders found in the top-level directory for batch processing.")

        for session_folder in session_folders:
            session_path = os.path.join(input_dir, session_folder)
            
            # Now go one level deeper to find the _track_ folders
            subfolders = [os.path.join(session_path, sub) for sub in os.listdir(session_path) 
                          if os.path.isdir(os.path.join(session_path, sub))]
            
            track_folders = []
            for subfolder in subfolders:
                track_folders += [d for d in os.listdir(subfolder) if '_track_' in d]
            
            if not track_folders:
                raise ValueError(f"No '_track_' folders found inside the session: {session_path}")
        
        return True  # Validation successful

    else:
        # Single session processing: Look for _track_ folders directly in the input directory
        track_folders = [d for d in os.listdir(input_dir) if '_track_' in d]
        if not track_folders:
            raise ValueError("No '_track_' folders found in the input directory.")
        return True
    
# Configure logging
logging.basicConfig(
    filename='ntf_app.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG
)
  
# Signal class for multithreading updates
class WorkerSignals(QObject):
    progress = pyqtSignal(int)
    message = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal()
    update_sd_threshold = pyqtSignal(str)
    update_k_value = pyqtSignal(int)
    plot_paths = pyqtSignal(list)

class Worker(QObject):
    def __init__(self, parent, input_dir, output_dir, fps):
        super().__init__()
        self.parent = parent
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.fps = fps
        self.signals = WorkerSignals()
        self.total_tracks = 0  # Keep track of the total number of tracks
        self.plot_files = []  # List to store plot file paths
        self.file_lock = Lock()  # Global lock to prevent concurrent access to files

    def run(self):
        try:
            self.signals.message.emit("Starting processing...")
            # Determine processing mode
            is_batch = self.parent.batch_checkbox.isChecked()
            if is_batch:
                # Batch processing mode
                parent_dirs = [os.path.join(self.input_dir, d) for d in os.listdir(self.input_dir)
                               if os.path.isdir(os.path.join(self.input_dir, d))]
            else:
                # Single directory processing
                parent_dirs = [self.input_dir]

            total_dirs = len(parent_dirs)
            for idx, parent_dir in enumerate(parent_dirs):
                self.signals.message.emit(f"Processing directory {idx + 1}/{total_dirs}: {parent_dir}")
                progress = int((idx / total_dirs) * 100)
                self.signals.progress.emit(progress)
                self.process_directory(parent_dir, self.output_dir, self.fps, idx)
            self.signals.progress.emit(100)
            self.signals.message.emit("Processing completed.")
            self.signals.plot_paths.emit(self.plot_files)  # Send plot paths to GUI
            self.signals.finished.emit()
        except Exception as e:
            # Get full traceback details
            error_message = ''.join(traceback.format_exception(None, e, e.__traceback__))
            logging.error("Error in Worker run:", exc_info=True)
            self.signals.error.emit(error_message)  # Emit detailed error signal to UI

    def process_directory(self, parent_dir, output_dir, fps, dir_idx):
        try:
            sd_values = []
            identifiers = []

            # Prepare output subdirectory automatically inside the parent folder for batch processing
            if self.parent.batch_checkbox.isChecked():
                 # Use the base name of the parent directory for more meaningful folder names
                output_subdir = os.path.join(parent_dir, "Output")
                os.makedirs(output_subdir, exist_ok=True)
            else:
                output_subdir = output_dir # For single directory processing

            # Check if we need to calculate SD
            if self.parent.calculate_sd_checkbox.isChecked():
                self.signals.message.emit("Calculating SD values...")
                # Recursively find subfolders with '_track_' in the name using os.walk
                subfolders = []
                for root, dirs, files in os.walk(parent_dir):
                    subfolders += [os.path.join(root, d) for d in dirs if '_track_' in d]
                # Log the found subfolders for debugging purposes
                self.signals.message.emit(f"Found track subfolders: {subfolders}")
                
                # Ensure that subfolders were found
                self.total_tracks = len(subfolders)
                if self.total_tracks == 0:
                    raise ValueError(f"No track folders found in {parent_dir}.")  # Raise an exception with a clear message

                total_subfolders = len(subfolders)
                with ThreadPoolExecutor() as executor:
                    futures = [executor.submit(self.calculate_sd, subfolder, identifiers, idx, total_subfolders)
                           for idx, subfolder in enumerate(subfolders)]
                    for future in futures:
                        future.result()

                if not identifiers:
                     raise ValueError("No valid SD values calculated. Please check your input data.")  # Raise another exception

                # Save SD Summary
                sd_summary_df = pd.DataFrame(identifiers)
                sd_summary_csv = os.path.join(output_subdir, 'SD_Summary.csv')
                sd_summary_df.to_csv(sd_summary_csv, index=False)
                self.signals.message.emit("SD Summary saved.")
            else:
                # Load existing SD values
                sd_summary_csv = os.path.join(output_subdir, 'SD_Summary.csv')
                if os.path.isfile(sd_summary_csv):
                    sd_summary_df = pd.read_csv(sd_summary_csv)
                    self.signals.message.emit("Loaded existing SD Summary.")
                else:
                    raise FileNotFoundError("SD_Summary.csv not found in the output directory.")

            # Check if we need to run clustering
            if self.parent.clustering_checkbox.isChecked():
                method = self.parent.cluster_method_combo.currentText()
                if method == "SD Threshold":
                    self.sd_threshold_clustering(sd_summary_df, output_subdir)
                else:
                    self.k_means_clustering(sd_summary_df, output_subdir)
            else:
                # Load existing clustering results
                clustering_csv = os.path.join(output_subdir, 'Clustering_Results.csv')
                if os.path.isfile(clustering_csv):
                    sd_summary_df = pd.read_csv(clustering_csv)
                    self.signals.message.emit("Loaded existing Clustering Results.")
                else:
                    raise FileNotFoundError("Clustering_Results.csv not found in the output directory.")

            # Check if we need to classify files
            if self.parent.classification_checkbox.isChecked():
                self.classify_files(sd_summary_df, output_subdir)

            # Check if we need to compute movement metrics
            if self.parent.compute_metrics_checkbox.isChecked():
                self.compute_movement_metrics(sd_summary_df, output_subdir, fps)
        except Exception as e:
            error_message = f"Error in process_directory: {str(e)}" if str(e) else "Unknown error in process_directory."
            logging.error(error_message, exc_info=True)
            self.signals.error.emit(error_message)
            raise

    from threading import Lock
    file_lock = Lock()  # Global lock to prevent concurrent access to files
    
    def calculate_sd(self, subfolder, identifiers, idx, total):
        try:
            identifier = os.path.basename(subfolder)
            # Adjusted to match your data structure
            track_file = os.path.join(subfolder, 'track.txt')
             # Ensure the file is accessed by only one thread at a time
            with self.file_lock:  # Using the instance variable file_lock
                if not os.path.isfile(track_file):
                    self.signals.message.emit(f"Missing file: {track_file}")
                    return
                with open(track_file, 'r') as file:
                    df = pd.read_csv(track_file)
            # Validate required columns
            if not {'X', 'Y'}.issubset(df.columns):
                self.signals.message.emit(f"Invalid format in {track_file}")
                return
            sd_x = df['X'].std()
            sd_y = df['Y'].std()
            euclidean_sd = np.sqrt(sd_x**2 + sd_y**2)
            identifiers.append({'ID': identifier, 'SD(X)': sd_x, 'SD(Y)': sd_y,
                                'EuclideanNorm_SD(X,Y)': euclidean_sd, 'Path': subfolder})
            progress = int((idx / total) * 100)
            self.signals.progress.emit(progress)
        except Exception as e:
            error_message = f"Error in calculate_sd for {subfolder}: {str(e) or repr(e)}"
            logging.error(error_message, exc_info=True)
            self.signals.error.emit(error_message)

    def sd_threshold_clustering(self, sd_summary_df, output_dir):
        try:
            self.signals.message.emit("Running SD Threshold Clustering...")
            # Get threshold
            if self.parent.auto_threshold_checkbox.isChecked():
                # Auto-thresholding
                threshold = self.auto_threshold(sd_summary_df['EuclideanNorm_SD(X,Y)'])
                self.signals.message.emit(f"Auto Threshold Calculated: {threshold:.4f}")
                # Emit a signal to update the SD threshold input field
                self.signals.update_sd_threshold.emit(f"{threshold:.4f}")
            else:
                try:
                    threshold = float(self.parent.sd_threshold_input.text())
                except ValueError:
                    self.signals.error.emit("Invalid SD Threshold value.")
                    return
            # Classify based on threshold
            sd_summary_df['Cluster'] = sd_summary_df['EuclideanNorm_SD(X,Y)'].apply(
                lambda x: 'Stationary' if x <= threshold else 'Non-Stationary')

            # Save clustering results
            clustering_csv = os.path.join(output_dir, 'Clustering_Results.csv')
            sd_summary_df.to_csv(clustering_csv, index=False)
            self.signals.message.emit("Clustering Results saved.")

            # Visualize clustering
            self.visualize_clusters(sd_summary_df, output_dir)
        except Exception as e:
            error_message = f"Error in sd_threshold_clustering: {str(e) or repr(e)}"
            logging.error(error_message, exc_info=True)
            self.signals.error.emit(error_message)
            raise

    def auto_threshold(self, sd_values):
        try:
            self.signals.message.emit("Calculating auto-threshold using first derivative...")
            # Sort the SD values
            sd_sorted = np.sort(sd_values)
            # Compute first derivative (differences between consecutive SD values)
            first_derivative = np.diff(sd_sorted)
            # Calculate relative increases
            relative_increases = first_derivative / sd_sorted[:-1]
            # Identify the index of the maximum relative increase
            max_increase_idx = np.argmax(relative_increases)
            # Set threshold at the SD value corresponding to this index
            threshold = sd_sorted[max_increase_idx]
            return threshold
        except Exception as e:
            error_message = f"Error in auto_threshold: {str(e) or repr(e)}"
            logging.error(error_message, exc_info=True)
            self.signals.error.emit(error_message)
            raise

    def k_means_clustering(self, sd_summary_df, output_dir):
        try:
            self.signals.message.emit("Running K-Means Clustering...")
            # Prepare data
            sd_values = sd_summary_df['EuclideanNorm_SD(X,Y)'].values.reshape(-1, 1)

            if len(sd_values) < 2:
                self.signals.error.emit("Not enough data points for K-Means clustering.")
                return

            # Get number of clusters
            if self.parent.k_auto_checkbox.isChecked():
                # Auto-detect optimal k using Silhouette Score
                silhouette_scores = []
                K = range(2, min(6, len(sd_values)))
                if len(K) < 2:
                    K = [2]
                for k in K:
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    labels = kmeans.fit_predict(sd_values)
                    score = silhouette_score(sd_values, labels)
                    silhouette_scores.append(score)
                optimal_k = K[np.argmax(silhouette_scores)]
                k = optimal_k
                self.signals.message.emit(f"Optimal number of clusters detected: {k}")
                # Emit a signal to update the k_input field
                self.signals.update_k_value.emit(k)
            else:
                k = self.parent.k_input.value()

            # K-Means clustering
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(sd_values)
            sd_summary_df['Cluster_Label'] = labels
            cluster_centers = kmeans.cluster_centers_.flatten()
            sorted_indices = np.argsort(cluster_centers)
            cluster_labels = ['Stationary', 'Non-Stationary'] + [f'Cluster {i+3}' for i in range(k - 2)]
            label_mapping = {sorted_indices[i]: cluster_labels[i] for i in range(k)}
            sd_summary_df['Cluster'] = sd_summary_df['Cluster_Label'].map(label_mapping)

            # Save clustering results
            clustering_csv = os.path.join(output_dir, 'Clustering_Results.csv')
            sd_summary_df.to_csv(clustering_csv, index=False)
            self.signals.message.emit("Clustering Results saved.")
            logging.info(f"Clustering Results saved at: {clustering_csv}")  # Log the path of Clustering.csv
            
            # Print the path to the console for debugging
            print(f"Clustering Results saved at: {clustering_csv}")

            # Visualize clustering
            self.visualize_clusters(sd_summary_df, output_dir)
        except Exception as e:
            error_message = f"Error in k_means_clustering: {str(e) or repr(e)}"
            logging.error(error_message, exc_info=True)
            self.signals.error.emit(error_message)
            raise
        
    def delete_stationary_files(self, input_dir, output_dir, *args, **kwargs):
        try:
            if self.parent.batch_checkbox.isChecked():
                # Batch mode: Iterate over session folders
                session_folders = [os.path.join(input_dir, d) for d in os.listdir(input_dir)
                                if os.path.isdir(os.path.join(input_dir, d))]
                for session_folder in session_folders:
                    output_subdir = os.path.join(session_folder, 'Output')
                    clustering_csv = os.path.join(output_subdir, 'Clustering_Results.csv')
                    if not os.path.isfile(clustering_csv):
                        self.signals.message.emit(f"Clustering_Results.csv not found in {output_subdir}. Skipping deletion for this session.")
                        continue  # Skip to the next session folder
                    self.perform_deletion(clustering_csv)
            else:
                # Single session mode
                output_subdir = os.path.join(input_dir, 'Output')
                clustering_csv = os.path.join(output_subdir, 'Clustering_Results.csv')
                if not os.path.isfile(clustering_csv):
                    self.signals.message.emit(f"Clustering_Results.csv not found in {output_subdir}. Skipping deletion.")
                    return  # Exit the function if the file is not found
                self.perform_deletion(clustering_csv)
            self.signals.message.emit("Stationary files deleted successfully.")
        except Exception as e:
            error_message = f"Error in delete_stationary_files: {str(e) or repr(e)}"
            logging.error(error_message, exc_info=True)
            self.signals.error.emit(error_message)

    def perform_deletion(self, clustering_csv):
        # Load clustering results
        clustering_df = pd.read_csv(clustering_csv)
        stationary_paths = clustering_df[clustering_df['Cluster'] == 'Stationary']['Path']
        # Delete each stationary directory or file
        for path in stationary_paths:
            if os.path.exists(path):
                shutil.rmtree(path)  # Use rmtree to delete directories
                logging.info(f"Deleted stationary folder: {path}")

    def visualize_clusters(self, sd_summary_df, output_dir):
        try:
            self.signals.message.emit("Generating interactive plot...")
            sd_summary_df['Short_ID'] = sd_summary_df['ID'].apply(lambda x: '_'.join(x.split('_')[-2:]))
            # Sort the dataframe by 'EuclideanNorm_SD(X,Y)'
            sd_summary_df_sorted = sd_summary_df.sort_values(by='EuclideanNorm_SD(X,Y)')
            sd_summary_df_sorted = sd_summary_df_sorted.reset_index(drop=True)
            sd_summary_df_sorted['Index'] = sd_summary_df_sorted.index + 1  # Start index from 1
            fig = px.scatter(sd_summary_df_sorted, x='Index', y='EuclideanNorm_SD(X,Y)',
                             color='Cluster', hover_data=['Short_ID', 'SD(X)', 'SD(Y)'])
            fig.update_layout(title='Clustering Results', xaxis_title='Sample Index',
                              yaxis_title='EuclideanNorm_SD(X,Y)', template='plotly_dark')
            # Save interactive plot
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_file = os.path.join(output_dir, f'Clustering_Results_{timestamp}.html')
            po.plot(fig, filename=plot_file, auto_open=False)
            self.signals.message.emit(f"Interactive plot saved at {plot_file}")
            # Add plot file to the list
            self.plot_files.append(plot_file)
        except Exception as e:
            error_message = f"Error in visualize_clusters: {str(e) or repr(e)}"
            logging.error(error_message, exc_info=True)
            self.signals.error.emit(error_message)

    def classify_files(self, sd_summary_df, output_dir):
        try:
            self.signals.message.emit("Classifying files...")
            # Customizable Output Structure
            stationary_dir = os.path.join(output_dir, 'stationary')
            non_stationary_dir = os.path.join(output_dir, 'non_stationary')
            unclassified_dir = os.path.join(output_dir, 'unclassified')
            os.makedirs(stationary_dir, exist_ok=True)
            os.makedirs(non_stationary_dir, exist_ok=True)
            os.makedirs(unclassified_dir, exist_ok=True)

            # Multithreading for file copying
            with ThreadPoolExecutor() as executor:
                futures = []
                for idx, row in sd_summary_df.iterrows():
                    futures.append(executor.submit(self.copy_file, row, stationary_dir,
                                                   non_stationary_dir, unclassified_dir))
                for future in futures:
                    future.result()
            self.signals.message.emit("File classification completed.")
        except Exception as e:
            error_message = f"Error in classify_files: {str(e) or repr(e)}"
            logging.error(error_message, exc_info=True)
            self.signals.error.emit(error_message)

    def copy_file(self, row, stationary_dir, non_stationary_dir, unclassified_dir):
        try:
            src = row['Path']
            if row['Cluster'] == 'Stationary':
                dst = os.path.join(stationary_dir, os.path.basename(src))
            elif row['Cluster'] == 'Non-Stationary':
                dst = os.path.join(non_stationary_dir, os.path.basename(src))
            else:
                dst = os.path.join(unclassified_dir, os.path.basename(src))
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        except Exception as e:
            error_message = f"Error copying {src} to {dst}: {str(e) or repr(e)}"
            logging.error(error_message, exc_info=True)
            self.signals.error.emit(error_message)

    def compute_movement_metrics(self, sd_summary_df, output_dir, fps):
        try:
            self.signals.message.emit("Computing movement metrics...")
            metrics = []
            total_tracks = len(sd_summary_df)
            for idx, row in sd_summary_df.iterrows():
                if row['Cluster'] != 'Non-Stationary':
                    continue  # Only compute for non-stationary tracks
                track_file = os.path.join(row['Path'], 'track.txt')
                try:
                    df = pd.read_csv(track_file)
                    # Ensure 'X' and 'Y' columns are present
                    if not {'X', 'Y'}.issubset(df.columns):
                        self.signals.message.emit(f"Missing 'X' or 'Y' in {track_file}")
                        continue
                    # Compute metrics
                    speed = self.compute_speed(df, fps)
                    displacement = self.compute_displacement(df)
                    path_length = self.compute_path_length(df)
                    if path_length == 0:
                        straightness = 0
                    else:
                        straightness = displacement / path_length
                    metrics.append({
                        'ID': row['ID'],
                        'Speed': speed,
                        'Displacement': displacement,
                        'Path Length': path_length,
                        'Trajectory Straightness': straightness
                    })
                    progress = int((idx / total_tracks) * 100)
                    self.signals.progress.emit(progress)
                except Exception as e:
                    error_message = f"Error computing metrics for {track_file}: {str(e) or repr(e)}"
                    logging.error(error_message, exc_info=True)
                    self.signals.error.emit(error_message)

            # Save metrics to CSV
            metrics_df = pd.DataFrame(metrics)
            metrics_csv = os.path.join(output_dir, 'Movement_Metrics.csv')
            metrics_df.to_csv(metrics_csv, index=False)
            self.signals.message.emit(f"Movement metrics saved at {metrics_csv}")
        except Exception as e:
            error_message = f"Error in compute_movement_metrics: {str(e) or repr(e)}"
            logging.error(error_message, exc_info=True)
            self.signals.error.emit(error_message)

    def compute_speed(self, df, fps):
        # Compute speed as the mean of the Euclidean distances between consecutive points
        distances = np.sqrt(np.diff(df['X'])**2 + np.diff(df['Y'])**2)
        speed = np.mean(distances) * fps
        return speed

    def compute_displacement(self, df):
        # Compute displacement as the Euclidean distance between the first and last point
        dx = df['X'].iloc[-1] - df['X'].iloc[0]
        dy = df['Y'].iloc[-1] - df['Y'].iloc[0]
        displacement = np.sqrt(dx**2 + dy**2)
        return displacement

    def compute_path_length(self, df):
        # Compute path length as the sum of Euclidean distances between consecutive points
        distances = np.sqrt(np.diff(df['X'])**2 + np.diff(df['Y'])**2)
        path_length = np.sum(distances)
        return path_length

class ClassificationWorker(QObject):
    def __init__(self, input_dir, output_dir, thresholds, auto_classification, batch_processing, dry_run):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.thresholds = thresholds
        self.auto_classification = auto_classification
        self.batch_processing = batch_processing
        self.dry_run = dry_run  # Ensure dry_run is set
        self.signals = WorkerSignals()
        self.file_lock = Lock()  # Global lock to prevent concurrent access to files

    def run(self):
        try:
            # Log the classification settings
            self.log_classification_settings()

            self.signals.message.emit("Starting classification...")
            self.process_directory()
            self.signals.progress.emit(100)
            self.signals.message.emit("Classification completed.")
            self.signals.finished.emit()
        except Exception as e:
            error_message = f"Error during classification: {str(e) or repr(e)}"
            logging.error(error_message, exc_info=True)
            self.signals.error.emit(error_message)
            self.signals.finished.emit()

    def log_classification_settings(self):
        if self.auto_classification:
            thresholds = {
                'eccentricity': 0.8,
                'solidity': 0.9,
                'aspect_ratio': 1.5,
                'circularity': 0.5
            }
        else:
            thresholds = self.thresholds

        settings = {
            'auto_classification': self.auto_classification,
            'thresholds': thresholds
        }
        logging.info(f"Classification settings: {settings}")
        self.signals.message.emit(f"Classification settings: {settings}")

    def process_directory(self):
        # Initialize results list
        results = []

        if self.batch_processing:
            # Get session folders in the input directory (top-level folders)
            session_folders = [os.path.join(self.input_dir, d) for d in os.listdir(self.input_dir)
                            if os.path.isdir(os.path.join(self.input_dir, d))]

            for session_folder in session_folders:
                # Recursive search for all _track_ subfolders inside each session folder
                subfolders = [os.path.join(root, d) for root, dirs, _ in os.walk(session_folder)
                            for d in dirs if '_track_' in d]
                # Ensure subfolders is defined before proceeding
                if not subfolders:
                    self.signals.message.emit(f"No '_track_' subfolders found in {session_folder}.")
                    continue
                
                # Create output directory inside each session folder
                output_subdir = os.path.join(session_folder, "Output")
                os.makedirs(output_subdir, exist_ok=True)  # Create the Output directory at the correct level

                # Process each _track_ subfolder
                total_folders = len(subfolders)
                for idx, subfolder in enumerate(subfolders):
                    track_tif = os.path.join(subfolder, 'track.tif')
                    if os.path.isfile(track_tif):
                        classification = self.process_image_stack(track_tif)  # Process the track.tif file
                        if classification:
                            
                            # In dry run mode, append results instead of moving files
                            results.append({'Folder': subfolder, 'Classification': classification})
                            
                            # Copy folder to output directory (inside the session folder)
                            if not self.dry_run:
                                self.copy_folder(subfolder, classification, output_subdir)

                            # Append results to the list
                            results.append({'Folder': subfolder, 'Classification': classification})
                    else:
                        self.signals.message.emit(f"track.tif not found in {subfolder}")

                    # Update progress
                    progress = int(((idx + 1) / total_folders) * 100)
                    self.signals.progress.emit(progress)

                # Save results to CSV at the end
                results_df = pd.DataFrame(results)
                csv_path = os.path.join(self.output_dir, 'classification_dry_run_results.csv' if self.dry_run else 'classification_results.csv')
                results_df.to_csv(csv_path, index=False)
                self.signals.message.emit(f"Results saved to {csv_path}")
        else:
            # Single directory mode: process the input directory like a batch
            subfolders = [os.path.join(self.input_dir, d) for d in os.listdir(self.input_dir)
                        if os.path.isdir(os.path.join(self.input_dir, d)) and '_track_' in d]
            
            if not subfolders:
                self.signals.message.emit(f"No '_track_' subfolders found in {self.input_dir}.")
                return

            total_folders = len(subfolders)
            for idx, subfolder in enumerate(subfolders):
                track_tif = os.path.join(subfolder, 'track.tif')
                if os.path.isfile(track_tif):
                    classification = self.process_image_stack(track_tif)
                    if classification:
                        output_subdir = os.path.join(self.input_dir, "Output")
                        os.makedirs(output_subdir, exist_ok=True)
                        self.copy_folder(subfolder, classification, output_subdir)
                        results.append({'Folder': subfolder, 'Classification': classification})
                else:
                    self.signals.message.emit(f"track.tif not found in {subfolder}")
                
                progress = int(((idx + 1) / total_folders) * 100)
                self.signals.progress.emit(progress)

        # Save classification results to CSV
        if results:
            results_df = pd.DataFrame(results)
            csv_path = os.path.join(self.output_dir, 'classification_results.csv')
            results_df.to_csv(csv_path, index=False)
            self.signals.message.emit(f"Results saved to {csv_path}")
        else:
            self.signals.message.emit("No classification results to save.")
        
    def process_image_stack(self, tif_path):
        try:
            # Ensure the file is accessed by only one thread at a time
            with self.file_lock:
                img = Image.open(tif_path)
                total_frames = img.n_frames  # Number of frames in the stack
                frame_indices = np.random.choice(range(total_frames), size=min(500, total_frames), replace=False)
                features_list = []

            for idx in frame_indices:
                img.seek(idx)  # Go to the frame at index idx
                frame = np.array(img)
                feature = self.extract_features(frame)  # Use the same feature extraction logic
                if feature:
                    features_list.append(feature)

            if features_list:
                # *** Start Modification ***
                # Convert features_list to DataFrame
                features_df = pd.DataFrame(features_list)
                # Compute average features
                avg_features = features_df.mean().to_dict()
                # *** End Modification ***
            
                classification = self.classify_object(avg_features)
                return classification
            else:
                self.signals.message.emit(f"No features extracted from {tif_path}")
                return None
        except Exception as e:
            error_message = f"Error processing image stack {tif_path}: {str(e) or repr(e)}"
            logging.error(error_message, exc_info=True)
            self.signals.error.emit(error_message)
            return None


    def extract_features(self, frame):
        try:
            # Check if the image has multiple channels (e.g., 3 for BGR)
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # Convert BGR to grayscale only if it has 3 channels
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                # If the image is already grayscale (1 channel), no need to convert
                gray = frame
                
            # Thresholding (invert if worms are darker)
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Assume the largest contour is the object
                contour = max(contours, key=cv2.contourArea)
                # Create a mask for the contour
                mask = np.zeros_like(gray)
                cv2.drawContours(mask, [contour], -1, color=255, thickness=-1)
                
                # *** Start Modification ***
                # Compute intensity features
                pixel_values = gray[mask == 255]
                mean_intensity = np.mean(pixel_values)
                std_intensity = np.std(pixel_values)
                min_intensity = np.min(pixel_values)
                max_intensity = np.max(pixel_values)
                # *** End Modification ***
                
                # Compute shape features
                shape_features = self.compute_shape_features(contour)

                # *** Start Modification ***
                # Combine features into a dictionary
                features = {
                    'mean_intensity': mean_intensity,
                    'std_intensity': std_intensity,
                    'min_intensity': min_intensity,
                    'max_intensity': max_intensity,
                    **shape_features
                }
                return features
                # *** End Modification ***
            else:
                return None
        except Exception as e:
            error_message = f"Error extracting features: {str(e) or repr(e)}"
            logging.error(error_message, exc_info=True)
            self.signals.error.emit(error_message)
            return None

    def compute_shape_features(self, contour):
        area = cv2.contourArea(contour)
        if area == 0:
            return None
        perimeter = cv2.arcLength(contour, True)

        # *** Start Modification ***
        # Compute additional shape descriptors
        # Compactness
        compactness = (perimeter ** 2) / (4 * np.pi * area) if area != 0 else 0
        # Aspect Ratio
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h != 0 else 0
        # Circularity
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter != 0 else 0
        # Solidity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area != 0 else 0
        # Eccentricity
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (center, axes, orientation) = ellipse
            majoraxis_length = max(axes)
            minoraxis_length = min(axes)
            eccentricity = np.sqrt(1 - (minoraxis_length / majoraxis_length) ** 2)
        else:
            eccentricity = 0
        # Compute curvature
        mean_curvature = self.compute_curvature(contour)

        # Combine features into a dictionary
        features = {
            'area': area,
            'perimeter': perimeter,
            'compactness': compactness,
            'aspect_ratio': aspect_ratio,
            'circularity': circularity,
            'solidity': solidity,
            'eccentricity': eccentricity,
            'mean_curvature': mean_curvature
        }
        return features
        # *** End Modification ***
        
    # *** Start Modification ***
    def compute_curvature(self, contour):
        curvature = []
        for i in range(len(contour)):
            prev_point = contour[i - 1][0]
            curr_point = contour[i][0]
            next_point = contour[(i + 1) % len(contour)][0]

            # Vectors
            vec1 = curr_point - prev_point
            vec2 = next_point - curr_point

            # Calculate angle between vectors
            angle = self.angle_between(vec1, vec2)
            curvature.append(angle)
        mean_curvature = np.mean(curvature)
        return mean_curvature

    def angle_between(self, v1, v2):
        # Calculate angle between two vectors
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        return np.degrees(angle)
    # *** End Modification ***

    def classify_object(self, features):
        # Retrieve feature values
        mean_intensity = features['mean_intensity']
        compactness = features['compactness']
        mean_curvature = features['mean_curvature']
        # Existing features
        eccentricity = features['eccentricity']
        solidity = features['solidity']
        aspect_ratio = features['aspect_ratio']
        circularity = features['circularity']

        if self.auto_classification:
            # Default thresholds
            mean_intensity_thresh = 100
            compactness_thresh = 15
            mean_curvature_thresh = 30
            ecc_thresh = 0.8
            sol_thresh = 0.9
            ar_thresh = 1.5
            circ_thresh = 0.5
        else:
            mean_intensity_thresh = self.thresholds['mean_intensity']
            compactness_thresh = self.thresholds['compactness']
            mean_curvature_thresh = self.thresholds['mean_curvature']
            ecc_thresh = self.thresholds['eccentricity']
            sol_thresh = self.thresholds['solidity']
            ar_thresh = self.thresholds['aspect_ratio']
            circ_thresh = self.thresholds['circularity']

        # *** Start Modification ***
        # Classification rules incorporating new features
        if (mean_intensity < mean_intensity_thresh and compactness > compactness_thresh) or \
        (mean_curvature > mean_curvature_thresh):
            return 'worm'
        elif mean_intensity >= mean_intensity_thresh:
            return 'artifact'
        else:
            return 'unknown'
        # *** End Modification ***

    def copy_folder(self, source_folder, classification, output_subdir):
        try:
            if self.dry_run:
            # In dry run mode, log the intended operation only
                self.signals.message.emit(f"[Dry Run] Would classify {source_folder} as '{classification}'")
                return  # Exit without copying
        
            # Copy the entire source_folder
            dest_folder = os.path.join(output_subdir, classification, os.path.basename(source_folder))
            if os.path.exists(dest_folder):
                shutil.rmtree(dest_folder)
                shutil.copytree(source_folder, dest_folder)
        except Exception as e:
            error_message = f"Error copying folder {source_folder}: {str(e) or repr(e)}"
            logging.error(error_message, exc_info=True)
            self.signals.error.emit(error_message)
            
class NematodeTracksFilter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nematode Tracks Filter")
        self.setGeometry(100, 100, 900, 700)
        self.setStyleSheet("background-color: #2b2b2b; color: #f0f0f0;")
        self.history = []
        self.initUI()
        self.load_history()
        self.plot_files = []  # List to store plot file paths
        self.current_plot_index = 0  # Index to keep track of current plot

    def initUI(self):
        # Main widget and layout
        self.main_widget = QWidget()
        self.layout = QVBoxLayout()

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabBar::tab {
                background: #1e1e1e;
                color: #f0f0f0;
                padding: 10px;
            }
            QTabBar::tab:selected {
                background: #2b2b2b;
            }
            QTabWidget::pane {
                border: 1px solid #444444;
            }
        """)
        self.layout.addWidget(self.tabs)

        # Tab 1: Configuration
        self.tab1 = QWidget()
        self.tabs.addTab(self.tab1, "Configuration")
        self.init_tab1()

        # Tab 2: History
        self.tab2 = QWidget()
        self.tabs.addTab(self.tab2, "History")
        self.init_tab2()

        # Tab 3: Logs and Messages
        self.tab3 = QWidget()
        self.tabs.addTab(self.tab3, "Process Logs")
        self.init_tab3()

        # New Tab: Advanced Analysis
        self.tab_advanced_analysis = QWidget()
        self.tabs.addTab(self.tab_advanced_analysis, "Advanced Analysis")
        self.init_advanced_analysis_tab()

        # Set main widget and layout
        self.main_widget.setLayout(self.layout)
        self.setCentralWidget(self.main_widget)

        # Thread and Worker
        self.thread = None
        self.worker = None

        # Additional signal connections
        self.signals = WorkerSignals()
        self.signals.update_sd_threshold.connect(self.update_sd_threshold_input)
        self.signals.update_k_value.connect(self.update_k_input)

        # Initialize plot navigation buttons
        self.prev_button.setEnabled(False)
        self.next_button.setEnabled(False)

    def init_tab1(self):
        # Create the main layout for tab1
        self.tab1_layout = QVBoxLayout()

        # Create a splitter to divide the controls and viewing panel
        self.splitter = QSplitter(Qt.Vertical)

        # Top Controls Widget and Layout
        self.top_controls_widget = QWidget()
        self.top_controls_layout = QVBoxLayout()

        # Input Directory Selection
        self.input_label = QLabel("Input Directory:")
        self.input_label.setStyleSheet("color: #f0f0f0;")
        self.input_path = QLineEdit()
        self.input_button = QPushButton("Browse")
        self.input_button.clicked.connect(self.browse_input)
        self.input_layout = QHBoxLayout()
        self.input_layout.addWidget(self.input_label)
        self.input_layout.addWidget(self.input_path)
        self.input_layout.addWidget(self.input_button)
        self.top_controls_layout.addLayout(self.input_layout)

        # Output Directory Selection
        self.output_label = QLabel("Output Directory:")
        self.output_label.setStyleSheet("color: #f0f0f0;")
        self.output_path = QLineEdit()
        self.output_button = QPushButton("Browse")
        self.output_button.clicked.connect(self.browse_output)
        self.output_layout = QHBoxLayout()
        self.output_layout.addWidget(self.output_label)
        self.output_layout.addWidget(self.output_path)
        self.output_layout.addWidget(self.output_button)
        self.top_controls_layout.addLayout(self.output_layout)

        # Frame Rate Input
        self.fps_label = QLabel("Frame Rate (FPS):")
        self.fps_label.setStyleSheet("color: #f0f0f0;")
        self.fps_input = QLineEdit()
        self.fps_input.setText("10")
        self.fps_layout = QHBoxLayout()
        self.fps_layout.addWidget(self.fps_label)
        self.fps_layout.addWidget(self.fps_input)
        self.top_controls_layout.addLayout(self.fps_layout)

        # Processing Pipeline Selection
        self.pipeline_label = QLabel("Processing Pipeline:")
        self.pipeline_label.setStyleSheet("color: #f0f0f0; font-weight: bold;")
        self.top_controls_layout.addWidget(self.pipeline_label)

        self.pipeline_group = QGroupBox()
        self.pipeline_layout = QVBoxLayout()
        self.calculate_sd_checkbox = QCheckBox("Calculate SD")
        self.clustering_checkbox = QCheckBox("Run Clustering")
        self.classification_checkbox = QCheckBox("Classify Files")
        self.compute_metrics_checkbox = QCheckBox("Compute Movement Metrics")
        self.calculate_sd_checkbox.setChecked(True)
        self.clustering_checkbox.setChecked(True)
        self.classification_checkbox.setChecked(True)
        self.compute_metrics_checkbox.setChecked(True)
        self.pipeline_layout.addWidget(self.calculate_sd_checkbox)
        self.pipeline_layout.addWidget(self.clustering_checkbox)
        self.pipeline_layout.addWidget(self.classification_checkbox)
        self.pipeline_layout.addWidget(self.compute_metrics_checkbox)
        self.pipeline_group.setLayout(self.pipeline_layout)
        self.top_controls_layout.addWidget(self.pipeline_group)
        
        # Deletion Checkbox
        self.delete_stationary_checkbox = QCheckBox("Delete Stationary Files After Processing")
        self.delete_stationary_checkbox.setStyleSheet("color: #f0f0f0;")
        self.pipeline_layout.addWidget(self.delete_stationary_checkbox)
        
        # Connect the stateChanged signals to check for conflicts
        self.delete_stationary_checkbox.stateChanged.connect(self.check_for_conflicts)
        self.classification_checkbox.stateChanged.connect(self.check_for_conflicts)

        # Clustering Method Selection
        self.cluster_method_label = QLabel("Clustering Method:")
        self.cluster_method_label.setStyleSheet("color: #f0f0f0;")
        self.cluster_method_combo = QComboBox()
        self.cluster_method_combo.addItems(["SD Threshold", "K-Means"])
        self.cluster_method_combo.currentIndexChanged.connect(self.toggle_clustering_options)
        self.cluster_method_layout = QHBoxLayout()
        self.cluster_method_layout.addWidget(self.cluster_method_label)
        self.cluster_method_layout.addWidget(self.cluster_method_combo)
        self.top_controls_layout.addLayout(self.cluster_method_layout)

        # SD Threshold Input
        self.sd_threshold_label = QLabel("SD Threshold:")
        self.sd_threshold_label.setStyleSheet("color: #f0f0f0;")
        self.sd_threshold_input = QLineEdit()
        self.auto_threshold_checkbox = QCheckBox("Auto Threshold")
        self.auto_threshold_checkbox.setStyleSheet("color: #f0f0f0;")
        self.auto_threshold_checkbox.setChecked(True)
        self.sd_threshold_layout = QHBoxLayout()
        self.sd_threshold_layout.addWidget(self.sd_threshold_label)
        self.sd_threshold_layout.addWidget(self.sd_threshold_input)
        self.sd_threshold_layout.addWidget(self.auto_threshold_checkbox)
        self.top_controls_layout.addLayout(self.sd_threshold_layout)

        # K-Means Cluster Number Input
        self.k_label = QLabel("Number of Clusters (k):")
        self.k_label.setStyleSheet("color: #f0f0f0;")
        self.k_input = QSpinBox()
        self.k_input.setRange(2, 10)
        self.k_input.setValue(2)
        self.k_auto_checkbox = QCheckBox("Auto-detect k")
        self.k_auto_checkbox.setStyleSheet("color: #f0f0f0;")
        self.k_auto_checkbox.setChecked(True)
        self.k_layout = QHBoxLayout()
        self.k_layout.addWidget(self.k_label)
        self.k_layout.addWidget(self.k_input)
        self.k_layout.addWidget(self.k_auto_checkbox)
        self.top_controls_layout.addLayout(self.k_layout)
        self.k_label.hide()
        self.k_input.hide()
        self.k_auto_checkbox.hide()

        # Batch Processing Option
        self.batch_checkbox = QCheckBox("Enable Batch Processing")
        self.batch_checkbox.setStyleSheet("color: #f0f0f0;")
        self.batch_checkbox.stateChanged.connect(self.toggle_output_directory)  # Connect checkbox state change to method
        self.top_controls_layout.addWidget(self.batch_checkbox)

        # Process Monitoring
        self.progress_label = QLabel("Progress:")
        self.progress_label.setStyleSheet("color: #f0f0f0;")
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.top_controls_layout.addWidget(self.progress_label)
        self.top_controls_layout.addWidget(self.progress_bar)

        # Start Processing Button
        self.process_button = QPushButton("Start Processing")
        self.process_button.clicked.connect(self.start_processing)
        self.top_controls_layout.addWidget(self.process_button)

        # Set the layout and size policy for the top controls widget
        self.top_controls_widget.setLayout(self.top_controls_layout)
        self.top_controls_widget.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)

        # Add the top controls widget to the splitter
        self.splitter.addWidget(self.top_controls_widget)

        # Viewing Panel Widget and Layout
        self.viewing_panel_widget = QWidget()
        self.viewing_panel_layout = QVBoxLayout()

        # Viewing Panel Label
        self.viewing_panel_label = QLabel("Clustering Results:")
        self.viewing_panel_label.setStyleSheet("color: #f0f0f0; font-weight: bold;")
        self.viewing_panel_layout.addWidget(self.viewing_panel_label)

        # WebEngineView to display HTML plots
        self.web_view = QtWebEngineWidgets.QWebEngineView()
        self.web_view.setStyleSheet("background-color: #1e1e1e;")
        self.web_view.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.viewing_panel_layout.addWidget(self.web_view)

        # Navigation Buttons for Plots
        self.navigation_layout = QHBoxLayout()
        self.prev_button = QPushButton("Previous Plot")
        self.prev_button.clicked.connect(self.show_previous_plot)
        self.next_button = QPushButton("Next Plot")
        self.next_button.clicked.connect(self.show_next_plot)
        self.navigation_layout.addWidget(self.prev_button)
        self.navigation_layout.addWidget(self.next_button)
        self.viewing_panel_layout.addLayout(self.navigation_layout)

        # Set the layout for the viewing panel widget
        self.viewing_panel_widget.setLayout(self.viewing_panel_layout)
        self.viewing_panel_widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        # Add the viewing panel widget to the splitter
        self.splitter.addWidget(self.viewing_panel_widget)

        # Set initial sizes for the splitter
        self.splitter.setStretchFactor(0, 0)  # Top controls do not stretch
        self.splitter.setStretchFactor(1, 1)  # Viewing panel stretches to fill space

        # Add the splitter to the main tab layout
        self.tab1_layout.addWidget(self.splitter)

        self.tab1.setLayout(self.tab1_layout)
    
    def check_for_conflicts(self):
        if self.delete_stationary_checkbox.isChecked() and self.classification_checkbox.isChecked():
            QMessageBox.warning(
                self,
                "Conflict Detected",
                'Warning! Conflict Detected:\n\n'
                'The options "Delete Stationary Files After Processing" and "Classify Files" cannot be used simultaneously. '
                'Please select only one of these options to avoid conflicts in file handling during processing.'
            )
            # Optionally, you can uncheck one of the checkboxes to resolve the conflict
            # For example:
            # self.classification_checkbox.setChecked(False)
        
    # Add the toggle_output_directory method after init_tab1
    def toggle_output_directory(self, state):
        """
        Toggle the output directory field based on the batch processing checkbox.
        Disable the output directory field if batch processing is enabled.
        """
        if state == Qt.Checked:
            # Batch processing is enabled, disable the output path field and button
            self.output_path.setEnabled(False)
            self.output_button.setEnabled(False)
            self.log_message("Batch processing enabled. Output directory is automatically managed.")
        else:
            # Batch processing is disabled, enable the output path field and button
            self.output_path.setEnabled(True)
            self.output_button.setEnabled(True)
            self.log_message("Single directory processing mode. Please select an output directory.")
        
    def init_tab2(self):
        self.tab2_layout = QVBoxLayout()
        self.history_label = QLabel("History of Operations:")
        self.history_label.setStyleSheet("color: #f0f0f0;")
        self.history_list = QListWidget()
        self.load_history_button = QPushButton("Load Selected Configuration")
        self.load_history_button.clicked.connect(self.load_selected_history)
        self.tab2_layout.addWidget(self.history_label)
        self.tab2_layout.addWidget(self.history_list)
        self.tab2_layout.addWidget(self.load_history_button)
        self.tab2.setLayout(self.tab2_layout)

    def init_tab3(self):
        self.tab3_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background-color: #1e1e1e; color: #dcdcdc;")
        self.tab3_layout.addWidget(self.log_text)
        self.tab3.setLayout(self.tab3_layout)

    def init_advanced_analysis_tab(self):
        self.advanced_layout = QVBoxLayout()

        # Information Label
        info_label = QLabel("The functions in this tab use the input and output directories specified in the Configuration tab.")
        info_label.setStyleSheet("color: #f0f0f0;")
        info_label.setWordWrap(True)
        self.advanced_layout.addWidget(info_label)

        # Threshold Adjustment Controls
        self.threshold_group = QGroupBox("Threshold Adjustments")
        self.threshold_layout = QVBoxLayout()
        
        # Eccentricity
        self.ecc_label = QLabel("Eccentricity Threshold:")
        self.ecc_spin = QDoubleSpinBox()
        self.ecc_spin.setRange(0.0, 1.0)
        self.ecc_spin.setSingleStep(0.01)
        self.ecc_spin.setValue(0.8)
        self.ecc_layout = QHBoxLayout()
        self.ecc_layout.addWidget(self.ecc_label)
        self.ecc_layout.addWidget(self.ecc_spin)
        self.threshold_layout.addLayout(self.ecc_layout)
        
        # Solidity
        self.solidity_label = QLabel("Solidity Threshold:")
        self.solidity_spin = QDoubleSpinBox()
        self.solidity_spin.setRange(0.0, 1.0)
        self.solidity_spin.setSingleStep(0.01)
        self.solidity_spin.setValue(0.9)
        self.solidity_layout = QHBoxLayout()
        self.solidity_layout.addWidget(self.solidity_label)
        self.solidity_layout.addWidget(self.solidity_spin)
        self.threshold_layout.addLayout(self.solidity_layout)
        
        # Aspect Ratio
        self.aspect_ratio_label = QLabel("Aspect Ratio Threshold:")
        self.aspect_ratio_spin = QDoubleSpinBox()
        self.aspect_ratio_spin.setRange(0.0, 10.0)
        self.aspect_ratio_spin.setSingleStep(0.1)
        self.aspect_ratio_spin.setValue(1.5)
        self.aspect_ratio_layout = QHBoxLayout()
        self.aspect_ratio_layout.addWidget(self.aspect_ratio_label)
        self.aspect_ratio_layout.addWidget(self.aspect_ratio_spin)
        self.threshold_layout.addLayout(self.aspect_ratio_layout)
        
        # Circularity
        self.circularity_label = QLabel("Circularity Threshold:")
        self.circularity_spin = QDoubleSpinBox()
        self.circularity_spin.setRange(0.0, 1.0)
        self.circularity_spin.setSingleStep(0.01)
        self.circularity_spin.setValue(0.5)
        self.circularity_layout = QHBoxLayout()
        self.circularity_layout.addWidget(self.circularity_label)
        self.circularity_layout.addWidget(self.circularity_spin)
        self.threshold_layout.addLayout(self.circularity_layout)
        
        # *** Start Modification ***
        # Mean Intensity Threshold
        self.mean_intensity_label = QLabel("Mean Intensity Threshold:")
        self.mean_intensity_spin = QDoubleSpinBox()
        self.mean_intensity_spin.setRange(0.0, 255.0)
        self.mean_intensity_spin.setSingleStep(1.0)
        self.mean_intensity_spin.setValue(100.0)
        self.mean_intensity_layout = QHBoxLayout()
        self.mean_intensity_layout.addWidget(self.mean_intensity_label)
        self.mean_intensity_layout.addWidget(self.mean_intensity_spin)
        self.threshold_layout.addLayout(self.mean_intensity_layout)
        
        # Compactness Threshold
        self.compactness_label = QLabel("Compactness Threshold:")
        self.compactness_spin = QDoubleSpinBox()
        self.compactness_spin.setRange(0.0, 100.0)
        self.compactness_spin.setSingleStep(0.1)
        self.compactness_spin.setValue(15.0)
        self.compactness_layout = QHBoxLayout()
        self.compactness_layout.addWidget(self.compactness_label)
        self.compactness_layout.addWidget(self.compactness_spin)
        self.threshold_layout.addLayout(self.compactness_layout)
        
        # Mean Curvature Threshold
        self.mean_curvature_label = QLabel("Mean Curvature Threshold:")
        self.mean_curvature_spin = QDoubleSpinBox()
        self.mean_curvature_spin.setRange(0.0, 360.0)
        self.mean_curvature_spin.setSingleStep(1.0)
        self.mean_curvature_spin.setValue(30.0)
        self.mean_curvature_layout = QHBoxLayout()
        self.mean_curvature_layout.addWidget(self.mean_curvature_label)
        self.mean_curvature_layout.addWidget(self.mean_curvature_spin)
        self.threshold_layout.addLayout(self.mean_curvature_layout)
        # *** End Modification ***
        
        # Auto Threshold Checkbox
        self.auto_threshold_checkbox_class = QCheckBox("Use Auto Classification")
        self.auto_threshold_checkbox_class.setChecked(True)
        self.threshold_layout.addWidget(self.auto_threshold_checkbox_class)

        self.threshold_group.setLayout(self.threshold_layout)
        self.advanced_layout.addWidget(self.threshold_group)
        
        # Batch Processing Checkbox for Advanced Analysis
        self.batch_checkbox_advanced = QCheckBox("Enable Batch Processing")
        self.batch_checkbox_advanced.setStyleSheet("color: #f0f0f0;")
        self.batch_checkbox_advanced.setChecked(False)  # Default to unchecked
        self.advanced_layout.addWidget(self.batch_checkbox_advanced)
        
        # Start Classification Button
        self.classify_button = QPushButton("Start Classification")
        self.classify_button.clicked.connect(self.start_classification)
        self.advanced_layout.addWidget(self.classify_button)

        # Progress Bar
        self.class_progress_bar = QProgressBar()
        self.class_progress_bar.setValue(0)  # Set default progress to zero
        self.advanced_layout.addWidget(self.class_progress_bar)

        # Log Text
        self.class_log_text = QTextEdit()
        self.class_log_text.setReadOnly(True)
        self.advanced_layout.addWidget(self.class_log_text)
        
        # Advanced Analysis History List
        self.advanced_history_list = QListWidget()  # Create list widget to display history
        self.advanced_layout.addWidget(self.advanced_history_list)
        
        # Dry Run Checkbox
        self.dry_run_checkbox = QCheckBox("Dry Run Mode (No file modifications)")
        self.dry_run_checkbox.setChecked(False)  # Default to unchecked
        self.advanced_layout.addWidget(self.dry_run_checkbox)
        
        # Load Advanced History Button
        self.load_advanced_history_button = QPushButton("Load Selected Advanced Configuration")
        self.load_advanced_history_button.clicked.connect(self.load_selected_advanced_history)
        self.advanced_layout.addWidget(self.load_advanced_history_button)

        self.tab_advanced_analysis.setLayout(self.advanced_layout)

        
    # The new method `toggle_output_directory_advanced` here:
    def toggle_output_directory_advanced(self, state):
        """
        Toggle the output directory field based on the batch processing checkbox in Tab 4.
        Disable the output directory field if batch processing is enabled.
        """
        if state == Qt.Checked:
            # Batch processing is enabled, disable the output path field and button
            self.output_path.setEnabled(False)
            self.output_button.setEnabled(False)
            self.log_message("Batch processing enabled for Advanced Analysis. Output directory is automatically managed.")
        else:
            # Batch processing is disabled, enable the output path field and button
            self.output_path.setEnabled(True)
            self.output_button.setEnabled(True)
            self.log_message("Single directory processing mode for Advanced Analysis. Please select an output directory.")
            
    def start_classification(self):
        try:
            input_dir = self.input_path.text()
            
            # Determine if batch processing is enabled
            is_batch_mode = self.batch_checkbox_advanced.isChecked()
            
            # Capture the dry run checkbox state
            dry_run = self.dry_run_checkbox.isChecked()  # Capture checkbox state for dry run
            
             # Ensure the input directory exists before validating its structure
            if not os.path.isdir(input_dir):
                raise ValueError("Invalid input directory. Please select a valid directory.")
            
            # Validate input directory for batch or single session mode
            validate_input_directory(input_dir, is_batch_mode)

            # If batch processing is enabled, manage the output directory automatically
            if is_batch_mode:
                session_folder = os.path.dirname(input_dir)  # Gets the session folder
                output_dir = os.path.join(session_folder, "Output")  # Create Output directory inside the session folder
                self.log_message("Batch processing enabled. Output directory is automatically managed at: " + output_dir)
                os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist
            else:
                output_dir = self.output_path.text()
                if not os.path.isdir(output_dir):
                    raise ValueError("Invalid output directory. Please select a valid directory.")

            thresholds = {
                'mean_intensity': self.mean_intensity_spin.value(),
                'compactness': self.compactness_spin.value(),
                'mean_curvature': self.mean_curvature_spin.value(),
                'eccentricity': self.ecc_spin.value(),
                'solidity': self.solidity_spin.value(),
                'aspect_ratio': self.aspect_ratio_spin.value(),
                'circularity': self.circularity_spin.value()
            }
            auto_classification = self.auto_threshold_checkbox_class.isChecked()
            batch_processing = self.batch_checkbox_advanced.isChecked()  # Capture batch processing state

            # Log the classification settings
            settings = {
                'auto_classification': auto_classification,
                'thresholds': thresholds,
                'batch_processing': batch_processing  # Log the batch setting
            }
            logging.info(f"Classification settings: {settings}")
            self.log_message(f"Classification settings: {settings}")

            # Disable UI elements during processing
            self.classify_button.setEnabled(False)
            self.class_progress_bar.setValue(0)

            # Create worker and thread
            self.class_thread = QThread()
            self.class_worker = ClassificationWorker(input_dir, output_dir, thresholds, auto_classification, batch_processing, dry_run)
            self.class_worker.moveToThread(self.class_thread)

            # Connect signals
            self.class_thread.started.connect(self.class_worker.run)
            self.class_worker.signals.progress.connect(self.update_class_progress)
            self.class_worker.signals.message.connect(self.log_class_message)
            self.class_worker.signals.error.connect(self.log_class_error)
            self.class_worker.signals.finished.connect(self.class_thread.quit)
            self.class_worker.signals.finished.connect(self.class_worker.deleteLater)
            self.class_thread.finished.connect(self.class_thread.deleteLater)
             
            # Start the thread
            self.class_thread.start()

            # Re-enable the button when processing is done
            self.class_thread.finished.connect(lambda: self.classify_button.setEnabled(True))
        except Exception as e:
            error_message = f"Error in start_classification: {str(e) or repr(e)}"
            logging.error(error_message)
            self.log_class_error(error_message)
            QMessageBox.critical(self, "Error", str(e))

    def update_class_progress(self, value):
        self.class_progress_bar.setValue(value)

    def log_class_message(self, message):
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.class_log_text.append(f"[{timestamp}] {message}")
        logging.info(message)

    def log_class_error(self, message):
        if not message:
            message = "An unknown error occurred."
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.class_log_text.append(f"[{timestamp}] ERROR: {message}")
        logging.error(message, exc_info=True)
        QMessageBox.critical(self, "Error", message)

    def toggle_clustering_options(self):
        method = self.cluster_method_combo.currentText()
        if method == "SD Threshold":
            self.sd_threshold_label.show()
            self.sd_threshold_input.show()
            self.auto_threshold_checkbox.show()
            self.k_label.hide()
            self.k_input.hide()
            self.k_auto_checkbox.hide()
        else:
            self.sd_threshold_label.hide()
            self.sd_threshold_input.hide()
            self.auto_threshold_checkbox.hide()
            self.k_label.show()
            self.k_input.show()
            self.k_auto_checkbox.show()

    def update_sd_threshold_input(self, value):
        self.sd_threshold_input.setText(value)

    def update_k_input(self, value):
        self.k_input.setValue(value)

    def browse_input(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if directory:
            self.input_path.setText(directory)

    def browse_output(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_path.setText(directory)

    def start_processing(self):
        try:
            # Check for conflicting options
            if self.delete_stationary_checkbox.isChecked() and self.classification_checkbox.isChecked():
                QMessageBox.warning(
                    self,
                    "Conflict Detected",
                    'Warning! Conflict Detected:\n\n'
                    'The options "Delete Stationary Files After Processing" and "Classify Files" cannot be used simultaneously. '
                    'Please select only one of these options to avoid conflicts in file handling during processing.'
                )
                return  # Do not proceed with processing
            
            # Validate inputs
            input_dir = self.input_path.text()
            output_dir = self.output_path.text()
            fps = float(self.fps_input.text())
                    
            # Determine if batch processing is enabled
            is_batch_mode = self.batch_checkbox.isChecked()
            
            # Determine if deletion is enabled
            delete_stationary = self.delete_stationary_checkbox.isChecked()
            
             # Show warning dialog if deletion is enabled
            if delete_stationary:
                reply = QMessageBox.critical(
                    self, "Warning!",
                    "Warning! This action will permanently delete the stationary objects from the input directory and will clean the input directory to only include mobile worm tracks. Do you want to continue?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return  # Abort processing if user selects No

            # Validate input directory for batch or single session
            validate_input_directory(input_dir, is_batch_mode)

            # Additional validations
            if not os.path.isdir(input_dir):
                raise ValueError("Invalid input directory.")
            if not is_batch_mode and not os.path.isdir(output_dir):
                raise ValueError("Invalid output directory.")
            if fps <= 0:
                raise ValueError("Frame rate must be positive.")

            # Disable UI elements during processing
            self.process_button.setEnabled(False)
            self.progress_bar.setValue(0)

            # Save current configuration to history
            self.save_current_configuration()

            # Create worker and thread
            self.thread = QThread()
            self.worker = Worker(self, input_dir, output_dir, fps)
            self.worker.moveToThread(self.thread)

            # Connect signals
            self.thread.started.connect(self.worker.run)
            self.worker.signals.progress.connect(self.update_progress)
            self.worker.signals.message.connect(self.log_message)
            self.worker.signals.error.connect(self.log_error)
            self.worker.signals.finished.connect(self.thread.quit)

            # Update inputs
            self.worker.signals.update_sd_threshold.connect(self.update_sd_threshold_input)
            self.worker.signals.update_k_value.connect(self.update_k_input)
            self.worker.signals.plot_paths.connect(self.receive_plot_paths)
            
            # Add deletion step after clustering or classification
            if delete_stationary:
                if self.clustering_checkbox.isChecked():
                    # Connect the deletion function to the thread's finished signal
                    self.thread.finished.connect(lambda: self.worker.delete_stationary_files(input_dir, output_dir))
                else:
                    self.log_message("Clustering was not performed; skipping deletion of stationary files.")
            
            # Connect deleteLater signals after deletion
            self.worker.signals.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)

            # Start the thread
            self.thread.start()

            # Re-enable the button when processing is done
            self.thread.finished.connect(lambda: self.process_button.setEnabled(True))
             
        except Exception as e:
            # Handle exceptions with logging and error message
            error_message = f"Error in start_processing: {str(e) or repr(e)}"
            logging.error(error_message)
            self.log_error(error_message)
            QMessageBox.critical(self, "Error", str(e))

    def receive_plot_paths(self, plot_files):
        self.plot_files = plot_files
        if self.plot_files:
            self.current_plot_index = 0
            self.load_plot(self.current_plot_index)
            if len(self.plot_files) > 1:
                self.next_button.setEnabled(True)
        else:
            self.web_view.setHtml("<h3>No plots available to display.</h3>")

    def load_plot(self, index):
        plot_file = self.plot_files[index]
        url = QUrl.fromLocalFile(os.path.abspath(plot_file))
        self.web_view.load(url)
        self.update_navigation_buttons()

    def show_next_plot(self):
        if self.current_plot_index < len(self.plot_files) - 1:
            self.current_plot_index += 1
            self.load_plot(self.current_plot_index)

    def show_previous_plot(self):
        if self.current_plot_index > 0:
            self.current_plot_index -= 1
            self.load_plot(self.current_plot_index)

    def update_navigation_buttons(self):
        self.prev_button.setEnabled(self.current_plot_index > 0)
        self.next_button.setEnabled(self.current_plot_index < len(self.plot_files) - 1)

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def log_message(self, message):
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.log_text.append(f"[{timestamp}] {message}")
        logging.info(message)

    def log_error(self, message):
        if not message:
            message = "An unknown error occurred."
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.log_text.append(f"[{timestamp}] ERROR: {message}")
        logging.error(message, exc_info=True)
        error_details = traceback.format_exc()
        QMessageBox.critical(self, "Error", f"{message}\n\nDetails:\n{error_details}")

    def save_current_configuration(self):
        config = {
            'input_directory': self.input_path.text(),
            'output_directory': self.output_path.text(),
            'fps': self.fps_input.text(),
            'batch_processing': self.batch_checkbox.isChecked(),
            'clustering_method': self.cluster_method_combo.currentText(),
            'sd_threshold': self.sd_threshold_input.text(),
            'auto_sd_threshold': self.auto_threshold_checkbox.isChecked(),
            'k_value': self.k_input.value(),
            'auto_k': self.k_auto_checkbox.isChecked(),
            'pipeline': {
                'calculate_sd': self.calculate_sd_checkbox.isChecked(),
                'clustering': self.clustering_checkbox.isChecked(),
                'classification': self.classification_checkbox.isChecked(),
                'compute_metrics': self.compute_metrics_checkbox.isChecked()
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        self.history.append(config)
        self.save_history()
        self.update_history_list()

    def save_history(self):
        history_file = os.path.join(os.getcwd(), '.ntf_history.json')
        with open(history_file, 'w') as f:
            json.dump(self.history, f)
        print(f"History saved to {history_file}")  # Shows saved history path
    
    def save_advanced_history(self):
        advanced_history_file = os.path.join(os.getcwd(), '.ntf_advanced_history.json')
        with open(advanced_history_file, 'w') as f:
            json.dump(self.advanced_history, f)
        print(f"Advanced Analysis history saved to {advanced_history_file}")

    def load_history(self):
        history_file = os.path.join(os.getcwd(), '.ntf_history.json')
        if os.path.isfile(history_file):
            with open(history_file, 'r') as f:
                self.history = json.load(f)
            self.update_history_list()
        
        advanced_history_file = os.path.join(os.getcwd(), '.ntf_advanced_history.json')
        if os.path.isfile(advanced_history_file):
            with open(advanced_history_file, 'r') as f:
                self.advanced_history = json.load(f)
            self.update_advanced_history_list()
        print("History updated")
    
    def update_history_list(self):
        self.history_list.clear()
        for idx, config in enumerate(self.history):
            item_text = f"{idx + 1}: {config['timestamp']} - {config['input_directory']}"
            self.history_list.addItem(item_text)
            
    def update_advanced_history_list(self):
        self.advanced_history_list.clear()
        for idx, config in enumerate(self.advanced_history):
            item_text = f"{idx + 1}: {config['timestamp']} - {config['input_directory']}"
            self.advanced_history_list.addItem(item_text)
    
    def save_current_advanced_configuration(self):
        config = {
            'input_directory': self.input_path.text(),
            'output_directory': self.output_path.text(),
            'thresholds': {
                'mean_intensity': self.mean_intensity_spin.value(),
                'compactness': self.compactness_spin.value(),
                'mean_curvature': self.mean_curvature_spin.value(),
                'eccentricity': self.ecc_spin.value(),
                'solidity': self.solidity_spin.value(),
                'aspect_ratio': self.aspect_ratio_spin.value(),
                'circularity': self.circularity_spin.value()
            },
            'auto_classification': self.auto_threshold_checkbox_class.isChecked(),
            'batch_processing': self.batch_checkbox_advanced.isChecked(),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        self.advanced_history.append(config)
        self.save_advanced_history()
        self.update_advanced_history_list()
    
    def load_selected_advanced_history(self):
        selected_items = self.advanced_history_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Selection Error", "No configuration selected.")
            return
        idx = self.advanced_history_list.row(selected_items[0])
        config = self.advanced_history[idx]
        self.input_path.setText(config['input_directory'])
        self.output_path.setText(config['output_directory'])
        self.mean_intensity_spin.setValue(config['thresholds']['mean_intensity'])
        self.compactness_spin.setValue(config['thresholds']['compactness'])
        self.mean_curvature_spin.setValue(config['thresholds']['mean_curvature'])
        self.ecc_spin.setValue(config['thresholds']['eccentricity'])
        self.solidity_spin.setValue(config['thresholds']['solidity'])
        self.aspect_ratio_spin.setValue(config['thresholds']['aspect_ratio'])
        self.circularity_spin.setValue(config['thresholds']['circularity'])
        self.auto_threshold_checkbox_class.setChecked(config['auto_classification'])
        self.batch_checkbox_advanced.setChecked(config['batch_processing'])
        QMessageBox.information(self, "Advanced Configuration Loaded", "Advanced configuration loaded successfully.")

    def load_selected_history(self):
        selected_items = self.history_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Selection Error", "No configuration selected.")
            return
        idx = self.history_list.row(selected_items[0])
        config = self.history[idx]
        self.input_path.setText(config['input_directory'])
        self.output_path.setText(config['output_directory'])
        self.fps_input.setText(config['fps'])
        self.batch_checkbox.setChecked(config['batch_processing'])
        self.cluster_method_combo.setCurrentText(config['clustering_method'])
        self.sd_threshold_input.setText(config['sd_threshold'])
        self.auto_threshold_checkbox.setChecked(config['auto_sd_threshold'])
        self.k_input.setValue(config['k_value'])
        self.k_auto_checkbox.setChecked(config['auto_k'])
        self.calculate_sd_checkbox.setChecked(config['pipeline']['calculate_sd'])
        self.clustering_checkbox.setChecked(config['pipeline']['clustering'])
        self.classification_checkbox.setChecked(config['pipeline']['classification'])
        self.compute_metrics_checkbox.setChecked(config['pipeline']['compute_metrics'])
        QMessageBox.information(self, "Configuration Loaded", "Configuration loaded successfully.")

def main():
    app = QApplication(sys.argv)
    window = NematodeTracksFilter()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
