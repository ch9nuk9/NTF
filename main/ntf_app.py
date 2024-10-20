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
            error_message = f"Error in run: {str(e)}"
            logging.error(error_message, exc_info=True)
            self.signals.error.emit(error_message)
            self.signals.finished.emit()

    def process_directory(self, parent_dir, output_dir, fps, dir_idx):
        try:
            sd_values = []
            identifiers = []

            # Prepare output subdirectory for this dataset if batch processing
            if self.parent.batch_checkbox.isChecked():
                output_subdir = os.path.join(output_dir, f"dataset_{dir_idx+1}")
                os.makedirs(output_subdir, exist_ok=True)
            else:
                output_subdir = output_dir

            # Check if we need to calculate SD
            if self.parent.calculate_sd_checkbox.isChecked():
                self.signals.message.emit("Calculating SD values...")
                # Traverse subfolders and calculate SD
                subfolders = [os.path.join(parent_dir, d) for d in os.listdir(parent_dir)
                              if os.path.isdir(os.path.join(parent_dir, d))]
                # Filter subfolders to only include those ending with '_track_*'
                subfolders = [sf for sf in subfolders if os.path.basename(sf).startswith('_track_') or '_track_' in os.path.basename(sf)]

                self.total_tracks = len(subfolders)
                if self.total_tracks == 0:
                    self.signals.error.emit("No track folders found ending with '_track_*'.")
                    return

                total_subfolders = len(subfolders)
                with ThreadPoolExecutor() as executor:
                    futures = []
                    for idx, subfolder in enumerate(subfolders):
                        futures.append(executor.submit(self.calculate_sd, subfolder, identifiers, idx, total_subfolders))
                    for future in futures:
                        future.result()

                if not identifiers:
                    self.signals.error.emit("No valid SD values calculated. Please check your input data.")
                    return

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
            error_message = f"Error in process_directory: {str(e)}"
            logging.error(error_message, exc_info=True)
            self.signals.error.emit(error_message)
            raise

    def calculate_sd(self, subfolder, identifiers, idx, total):
        try:
            identifier = os.path.basename(subfolder)
            # Adjusted to match your data structure
            track_file = os.path.join(subfolder, 'track.txt')
            if not os.path.isfile(track_file):
                self.signals.message.emit(f"Missing file: {track_file}")
                return
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
            error_message = f"Error in calculate_sd for {subfolder}: {str(e)}"
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
            error_message = f"Error in sd_threshold_clustering: {str(e)}"
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
            error_message = f"Error in auto_threshold: {str(e)}"
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

            # Visualize clustering
            self.visualize_clusters(sd_summary_df, output_dir)
        except Exception as e:
            error_message = f"Error in k_means_clustering: {str(e)}"
            logging.error(error_message, exc_info=True)
            self.signals.error.emit(error_message)
            raise

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
            error_message = f"Error in visualize_clusters: {str(e)}"
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
            error_message = f"Error in classify_files: {str(e)}"
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
            error_message = f"Error copying {src} to {dst}: {str(e)}"
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
                    error_message = f"Error computing metrics for {track_file}: {str(e)}"
                    logging.error(error_message, exc_info=True)
                    self.signals.error.emit(error_message)

            # Save metrics to CSV
            metrics_df = pd.DataFrame(metrics)
            metrics_csv = os.path.join(output_dir, 'Movement_Metrics.csv')
            metrics_df.to_csv(metrics_csv, index=False)
            self.signals.message.emit(f"Movement metrics saved at {metrics_csv}")
        except Exception as e:
            error_message = f"Error in compute_movement_metrics: {str(e)}"
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
    def __init__(self, input_dir, output_dir, thresholds, auto_classification):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.thresholds = thresholds
        self.auto_classification = auto_classification
        self.signals = WorkerSignals()

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
            error_message = f"Error during classification: {str(e)}"
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
        subfolders = [os.path.join(self.input_dir, d) for d in os.listdir(self.input_dir)
                      if os.path.isdir(os.path.join(self.input_dir, d))]
        total_folders = len(subfolders)
        results = []

        for idx, subfolder in enumerate(subfolders):
            # Adjusted path to 'track.avi' inside the 'output' subdirectory
            track_video = os.path.join(subfolder, 'output', 'track.avi')
            if os.path.isfile(track_video):
                classification = self.process_video(track_video)
                if classification:
                    results.append({'Folder': subfolder, 'Classification': classification})
                    # Copy the folder to the output directory
                    self.copy_folder(subfolder, classification)
            else:
                self.signals.message.emit(f"track.avi not found in {subfolder}")
            progress = int(((idx + 1) / total_folders) * 100)
            self.signals.progress.emit(progress)

        # Save results to CSV
        results_df = pd.DataFrame(results)
        csv_path = os.path.join(self.output_dir, 'classification_results.csv')
        results_df.to_csv(csv_path, index=False)
        self.signals.message.emit(f"Results saved to {csv_path}")

    def process_video(self, video_path):
        try:
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count == 0:
                self.signals.message.emit(f"No frames found in {video_path}")
                cap.release()
                return None
            total_frames = min(500, frame_count)
            frame_indices = np.random.choice(range(frame_count), size=total_frames, replace=False)
            features_list = []

            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    feature = self.extract_features(frame)
                    if feature:
                        features_list.append(feature)

            cap.release()

            if features_list:
                # Average the features
                avg_features = np.mean(features_list, axis=0)
                classification = self.classify_object(avg_features)
                return classification
            else:
                self.signals.message.emit(f"No features extracted from {video_path}")
                return None
        except Exception as e:
            error_message = f"Error processing video {video_path}: {str(e)}"
            logging.error(error_message, exc_info=True)
            self.signals.error.emit(error_message)
            return None

    def extract_features(self, frame):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Thresholding
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Assume the largest contour is the object
                contour = max(contours, key=cv2.contourArea)
                features = self.compute_shape_features(contour)
                return features
            else:
                return None
        except Exception as e:
            error_message = f"Error extracting features: {str(e)}"
            logging.error(error_message, exc_info=True)
            self.signals.error.emit(error_message)
            return None

    def compute_shape_features(self, contour):
        area = cv2.contourArea(contour)
        if area == 0:
            return None
        perimeter = cv2.arcLength(contour, True)
        # Circularity
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter != 0 else 0
        # Aspect Ratio
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h != 0 else 0
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
        return [eccentricity, solidity, aspect_ratio, circularity]

    def classify_object(self, features):
        eccentricity, solidity, aspect_ratio, circularity = features
        if self.auto_classification:
            # Default thresholds
            ecc_thresh = 0.8
            sol_thresh = 0.9
            ar_thresh = 1.5
            circ_thresh = 0.5
        else:
            ecc_thresh = self.thresholds['eccentricity']
            sol_thresh = self.thresholds['solidity']
            ar_thresh = self.thresholds['aspect_ratio']
            circ_thresh = self.thresholds['circularity']

        # Classification rules
        if eccentricity > ecc_thresh and solidity < sol_thresh:
            return 'worm'
        elif aspect_ratio > ar_thresh and circularity < circ_thresh:
            return 'worm'
        elif solidity > sol_thresh and abs(aspect_ratio - 1) < 0.1:
            return 'artifact'
        else:
            return 'unknown'

    def copy_folder(self, source_folder, classification):
        try:
            # Copy the entire source_folder
            dest_folder = os.path.join(self.output_dir, classification, os.path.basename(source_folder))
            if os.path.exists(dest_folder):
                shutil.rmtree(dest_folder)
            shutil.copytree(source_folder, dest_folder)
        except Exception as e:
            error_message = f"Error copying folder {source_folder}: {str(e)}"
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
        self.tabs.addTab(self.tab3, "Logs & Messages")
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
        # Auto Threshold Checkbox
        self.auto_threshold_checkbox_class = QCheckBox("Use Auto Classification")
        self.auto_threshold_checkbox_class.setChecked(True)
        self.threshold_layout.addWidget(self.auto_threshold_checkbox_class)

        self.threshold_group.setLayout(self.threshold_layout)
        self.advanced_layout.addWidget(self.threshold_group)

        # Start Classification Button
        self.classify_button = QPushButton("Start Classification")
        self.classify_button.clicked.connect(self.start_classification)
        self.advanced_layout.addWidget(self.classify_button)

        # Progress Bar
        self.class_progress_bar = QProgressBar()
        self.advanced_layout.addWidget(self.class_progress_bar)

        # Log Text
        self.class_log_text = QTextEdit()
        self.class_log_text.setReadOnly(True)
        self.advanced_layout.addWidget(self.class_log_text)

        self.tab_advanced_analysis.setLayout(self.advanced_layout)

    def start_classification(self):
        try:
            input_dir = self.input_path.text()
            output_dir = self.output_path.text()
            if not os.path.isdir(input_dir):
                raise ValueError("Invalid input directory.")
            if not os.path.isdir(output_dir):
                raise ValueError("Invalid output directory.")

            thresholds = {
                'eccentricity': self.ecc_spin.value(),
                'solidity': self.solidity_spin.value(),
                'aspect_ratio': self.aspect_ratio_spin.value(),
                'circularity': self.circularity_spin.value()
            }
            auto_classification = self.auto_threshold_checkbox_class.isChecked()

            # Log the classification settings
            settings = {
                'auto_classification': auto_classification,
                'thresholds': thresholds
            }
            logging.info(f"Classification settings: {settings}")
            self.log_message(f"Classification settings: {settings}")

            # Disable UI elements during processing
            self.classify_button.setEnabled(False)
            self.class_progress_bar.setValue(0)

            # Create worker and thread
            self.class_thread = QThread()
            self.class_worker = ClassificationWorker(input_dir, output_dir, thresholds, auto_classification)
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
            error_message = f"Error in start_classification: {str(e)}"
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
            # Validate inputs
            input_dir = self.input_path.text()
            output_dir = self.output_path.text()
            fps = float(self.fps_input.text())
            if not os.path.isdir(input_dir):
                raise ValueError("Invalid input directory.")
            if not os.path.isdir(output_dir):
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
            self.worker.signals.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)
            self.worker.signals.update_sd_threshold.connect(self.update_sd_threshold_input)
            self.worker.signals.update_k_value.connect(self.update_k_input)
            self.worker.signals.plot_paths.connect(self.receive_plot_paths)  # New connection

            # Start the thread
            self.thread.start()

            # Re-enable the button when processing is done
            self.thread.finished.connect(lambda: self.process_button.setEnabled(True))
        except Exception as e:
            error_message = f"Error in start_processing: {str(e)}"
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
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.log_text.append(f"[{timestamp}] ERROR: {message}")
        logging.error(message, exc_info=True)
        QMessageBox.critical(self, "Error", message)

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
        history_file = os.path.join(os.path.expanduser('~'), '.ntf_history.json')
        with open(history_file, 'w') as f:
            json.dump(self.history, f)

    def load_history(self):
        history_file = os.path.join(os.path.expanduser('~'), '.ntf_history.json')
        if os.path.isfile(history_file):
            with open(history_file, 'r') as f:
                self.history = json.load(f)
            self.update_history_list()

    def update_history_list(self):
        self.history_list.clear()
        for idx, config in enumerate(self.history):
            item_text = f"{idx + 1}: {config['timestamp']} - {config['input_directory']}"
            self.history_list.addItem(item_text)

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
