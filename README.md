# Nematode Tracks Filter (NTF)

A comprehensive tool for analyzing and filtering nematode movement tracks from SWC output data.

---

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation Requirements](#installation-requirements)
- [Installation Guide](#installation-guide)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Set Up the Conda Environment](#2-set-up-the-conda-environment)
  - [3. Configure the Autorun Script](#3-configure-the-autorun-script)
    - [For Linux and macOS Users](#for-linux-and-macos-users)
    - [For Windows Users](#for-windows-users)
- [Usage Instructions](#usage-instructions)
  - [Running the Application](#running-the-application)
  - [Tabs Overview](#tabs-overview)
    - [Configuration Tab](#configuration-tab)
    - [History Tab](#history-tab)
    - [Logs & Messages Tab](#logs--messages-tab)
    - [Advanced Analysis Tab](#advanced-analysis-tab)
- [Design Logic](#design-logic)
  - [Processing Pipeline](#processing-pipeline)
  - [Classification Methodology](#classification-methodology)
  - [Threshold Adjustments](#threshold-adjustments)
- [Performance and Accuracy](#performance-and-accuracy)
- [Best Practices](#best-practices)
  - [When to Use Advanced Analysis](#when-to-use-advanced-analysis)
- [Conclusion](#conclusion)
- [Frequently Asked Questions (FAQ)](#frequently-asked-questions-faq)
- [Contribution Guidelines](#contribution-guidelines)
- [License](#license)
- [Contact Information](#contact-information)
- [Repository Files](#repository-files)
  - [`ntf_app.py`](#ntf_apppy)
  - [`run_ntf.sh`](#run_ntfsh)
  - [`run_ntf.bat`](#run_ntfbat)
  - [`LICENSE`](#license-file)
- [Detailed Explanation on How to Use the Script](#detailed-explanation-on-how-to-use-the-script)
  - [Setting Up Your Data](#setting-up-your-data)
  - [Running the Application](#running-the-application-1)
  - [Using Advanced Analysis](#using-advanced-analysis)
  - [Tips for Effective Use](#tips-for-effective-use)
  - [Troubleshooting](#troubleshooting)
- [Additional Resources](#additional-resources)

---

## Introduction

The **Nematode Tracks Filter (NTF)** is a Python-based graphical user interface (GUI) application designed to process, analyze, and classify movement tracks of nematodes (worms) captured through microscopy and later processed through the simple worm cropper (SWC). The application facilitates the filtering of movement data, clustering of tracks based on movement characteristics, and classification of tracks to distinguish worms from artifacts or unknown entities.

---

## Features

- **Batch Processing:** Analyze multiple datasets in batch mode for efficient workflow.
- **Standard Deviation Calculation:** Compute the standard deviation (SD) of X and Y coordinates for each track.
- **Clustering Methods:**
  - **SD Threshold Clustering:** Automatically or manually set thresholds to cluster tracks as stationary or non-stationary.
  - **K-Means Clustering:** Utilize K-Means algorithm with options for auto-detecting the optimal number of clusters.
- **File Classification:** Automatically organize tracks into folders based on clustering results.
- **Movement Metrics Computation:** Calculate speed, displacement, path length, and trajectory straightness for non-stationary tracks.
- **Advanced Analysis:**
  - **Custom Threshold Adjustments:** Fine-tune classification thresholds for eccentricity, solidity, aspect ratio, and circularity.
  - **Use Auto Classification:** Enable or disable auto classification using default thresholds.
- **Interactive Visualization:** Generate and view interactive plots of clustering results within the application.
- **History Management:** Save and load previous configurations for reproducibility.
- **Detailed Logging:** Monitor processing steps, errors, and messages through an integrated logging system.

---

## Installation Requirements

To run the Nematode Tracks Filter application, ensure you have the following installed:

- **Anaconda or Miniconda (Python 3.6+)**

**Required Python Packages:**

- PyQt5
- numpy
- pandas
- scikit-learn
- plotly
- opencv-python (cv2)

---

## Installation Guide

### 1. Clone the Repository

- Open a GitBash/GitHub desktop or terminal (with admin privileges) and clone the repository from GitHub:

```bash
git clone https://github.com/ch9nuk9/NTF
```
- Navigate to the cloned directory

```
cd path/to/your/NTF/repository/destination (may require a D switch unless in C drive)
```

### 2. Set Up the Conda Environment

- Create a new conda environment named ntf_env (or choose your preferred name) and install the required packages.

```
conda create -y -n ntf_env python=3.12
Activate the environment:
conda activate ntf_env

```

- Install the required packages:

```
conda install -y pyqt numpy pandas scikit-learn plotly OpenCV
```

- Alternatively, you can use pip to install packages not available via conda:

```
pip install PyQt5 opencv-python
```

### 3. Configure the Autorun Script
- To simplify running the application, you can create a script that automatically activates the conda environment and launches the GUI.

##### For Linux and macOS Users
##### 1. Create the Shell Script

- In the nematode-tracks-filter directory, create a file named run_ntf.sh (generated by default and can be found already inside the repository):
```
touch run_ntf.sh
```
##### 2. Edit the Shell Script
- Open run_ntf.sh with a text editor and add the following content:

```
#!/bin/bash
# Exit on error
set -e
# Activate conda environment
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate ntf_env
# Run the Python script
python "$(dirname "$0")/ntf_app.py"
```
######  Explanation:

-Shebang Line (#!/bin/bash): Specifies that the script should be run in the Bash shell.
-Exit on Error (set -e): The script exits if any command fails.
######  Conda Activation:
-Retrieves the base path of your conda installation.
-Sources the conda script to enable the conda command.
-Activates the ntf_env environment.
-Run the Python Script: Uses $(dirname "$0") to get the directory of the script, ensuring it can be run from any location.

#####  3. Make the Script Executable
#
```
chmod +x run_ntf.sh
```

##### 4. Create a Desktop Shortcut (Optional)

###### For Linux (GNOME/KDE):
- Create a file named NematodeTracksFilter.desktop in your desktop directory:

```
touch ~/Desktop/NematodeTracksFilter.desktop
```

- Add the following content to the file:

```
[Desktop Entry]
Type=Application
Name=Nematode Tracks Filter
Exec=/path/to/nematode-tracks-filter/run_ntf.sh
Icon=utilities-terminal
Terminal=false
```
- Replace /path/to/nematode-tracks-filter with the actual path to your cloned repository.
- Make the desktop entry executable:

```
chmod +x ~/Desktop/NematodeTracksFilter.desktop
```

- Allow launching by right-clicking the file and selecting Allow Launching.

###### For macOS Users:
- Use Automator to create an application that runs the shell script.
- Open Automator and create a new Application.
- Add a Run Shell Script action.
- Paste the content of run_ntf.sh into the action.
- Save the application and place it in your Applications folder or Desktop.


##### For Windows Users

##### 1. Create a Batch File (run_ntf.bat)

- In the nematode-tracks-filter directory, create run_ntf.bat:

```
@echo off
REM Activate conda environment
call C:\ProgramData\Anaconda3\Scripts\activate.bat ntf_env
REM Run the Python script
python "%~dp0ntf_app.py"
```

- Note: Adjust the path to activate.bat if your Anaconda installation is in a different location.

##### 2. Create a Shortcut

- Right-click run_ntf.bat and select Create Shortcut.
- Place the shortcut on your Desktop for easy access.


## Usage Instructions
### 1. Running the Application
- Using the Shell Script (Linux/macOS)
- Navigate to the nematode-tracks-filter directory:

```
cd /path/to/nematode-tracks-filter
```

- Run the shell script:

```
./run_ntf.sh
```

- Alternatively, double-click the desktop shortcut if you created one.
- Using the Batch File (Windows)
- Double-click the run_ntf.bat file or its shortcut to launch the application.


### 2. Tabs Overview
#### Configuration Tab
- The Configuration tab is the primary interface for setting up your data processing pipeline.

##### Input Directory:
- Specify the directory containing your track data. Use the Browse button to select the folder.
- Note: The input directory should contain subfolders with track.txt files and track.avi videos.

##### Output Directory:

- Specify where processed data and results will be saved. Use the Browse button to select the folder.

##### Frame Rate (FPS):
- Enter the frame rate of your videos. Default is 10 FPS.

##### Processing Pipeline:

- Calculate SD: Compute the standard deviation of X and Y coordinates for each track.
- Run Clustering: Perform clustering based on the selected method.
- Classify Files: Organize tracks into folders based on clustering results.
- Compute Movement Metrics: Calculate movement metrics for non-stationary tracks.

##### Clustering Method:
- Choose between SD Threshold and K-Means clustering methods.

###### SD Threshold Clustering Options:
- SD Threshold: Set a custom threshold or enable Auto Threshold to let the application calculate it automatically.
###### K-Means Clustering Options:
- Number of Clusters (k): Set the number of clusters or enable Auto-detect k to let the application determine the optimal number.

##### Enable Batch Processing:

- Process multiple datasets by enabling batch mode. Place each dataset in a separate subdirectory within the input directory.

##### Process Monitoring:

- Progress Bar: Visual indication of processing progress.
- Start Processing Button: Begin the processing pipeline with the configured settings.

##### Viewing Panel:

- Displays interactive plots of clustering results.
- Navigation Buttons: Use Previous Plot and Next Plot to navigate through multiple plots.

#### History Tab
- The History tab allows you to manage and load previous configurations.

##### History of Operations:

- View a list of saved configurations with timestamps and input directories.

##### Load Selected Configuration:
- Select a configuration from the list and load it into the application.

#### Logs & Messages Tab
- The Logs & Messages tab provides detailed logging information.

##### Log Text Area:
- View processing messages, errors, and other logs generated during application use.

#### Advanced Analysis Tab
- The Advanced Analysis tab offers tools for fine-tuning classification when dealing with noisy datasets.

##### Threshold Adjustments:
- Eccentricity Threshold
- Solidity Threshold
- Aspect Ratio Threshold
- Circularity Threshold
- Adjust these thresholds to refine classification accuracy.

##### Use Auto Classification:

- Enable to use default thresholds.
- Disable to apply custom thresholds specified in the input fields.

##### Start Classification Button:

- Initiate the classification process using the configured thresholds.
##### Progress Bar and Log Text:
- Monitor the progress and view logs specific to the advanced analysis.

## Design Logic
### Processing Pipeline
The application follows a structured processing pipeline to analyze nematode tracks:

#### 1. Standard Deviation Calculation:

- Computes the SD of X and Y coordinates from track.txt files.
- Calculates the Euclidean norm of SD values to represent movement variability.

#### 2. Clustering:

##### SD Threshold Clustering:
- Classifies tracks as stationary or non-stationary based on SD thresholds.
- Auto-thresholding calculates an optimal threshold using the first derivative method.

##### K-Means Clustering:
- Groups tracks into k clusters based on movement characteristics.
- Option to auto-detect the optimal number of clusters using the Silhouette Score.

#### 3. File Classification:

Organizes tracks into folders (stationary, non_stationary, unclassified) based on clustering results.

#### 4. Movement Metrics Computation:

- Calculates speed, displacement, path length, and trajectory straightness for non-stationary tracks.

#### 5. Visualization:
- Generates interactive plots using Plotly to visualize clustering results.

### Classification Methodology
In the Advanced Analysis tab, the application classifies tracks as worm, artifact, or unknown based on shape features extracted from video frames.

#### Feature Extraction:

- Samples frames from track.avi videos.
- Extracts contours and computes shape features:

###### 1. Eccentricity: Measures elongation of the object.
###### 2. Solidity: Ratio of contour area to convex hull area.
###### 3. Aspect Ratio: Ratio of width to height of the bounding rectangle.
###### 4. Circularity: Describes how close the shape is to a perfect circle.
#
#
#### Classification Rules:

###### 1. Worm:
- High eccentricity and low solidity.
- High aspect ratio and low circularity.
###### 2. Artifact:
- High solidity and aspect ratio close to 1.
###### 3. Unknown:
- Does not meet criteria for worm or artifact.

#### Threshold Adjustments
- Users can adjust thresholds for shape features to fine-tune classification, especially in datasets with high noise and artifacts.
##### Auto Classification:
- Uses default thresholds: Eccentricity: 0.8, Solidity: 0.9, Aspect Ratio: 1.5, Circularity: 0.5
- Custom Thresholds: Disable auto classification to input custom values.

## Performance and Accuracy
### Classification Accuracy:
- In testing, the application correctly classified 114 out of 116 data files.
- Accuracy Level: Approximately 98.28%.
#### Interpretation:
- High accuracy indicates reliable performance in distinguishing worms from artifacts.
Misclassifications may occur due to overlapping feature characteristics between worms and artifacts.


## Best Practices
### When to Use Advanced Analysis
##### Recommended Use:
- Use the Advanced Analysis tab only when your dataset is very noisy and full of artifacts.
- Adjusting thresholds can help improve classification in challenging datasets.
Default Processing:
- For typical datasets, the standard processing pipeline in the Configuration tab suffices.
- Auto thresholds and default settings are optimized for general cases.
Threshold Adjustments:
- Carefully adjust thresholds incrementally.
- Monitor the impact on classification results to avoid overfitting or misclassification.

## Frequently Asked Questions (FAQ)
- Q1: Can I adjust the default thresholds used in auto classification?
A1: Yes, you can disable Use Auto Classification in the Advanced Analysis tab to input custom threshold values.

- Q2: What file formats are supported for input data?
A2: The application expects directories containing track.txt files with X and Y coordinates, and track.avi videos for advanced analysis.

- Q3: How do I interpret the interactive plots generated?
A3: The plots visualize clustering results, showing the distribution of tracks based on movement characteristics. Hover over data points for detailed information.

- Q4: Can I process multiple datasets at once?
A4: Yes, enable Batch Processing in the Configuration tab to process multiple datasets located in subdirectories of the input directory.

- Q5: How do I handle misclassified tracks?
A5: You can adjust classification thresholds in the Advanced Analysis tab to refine the classification. Analyze misclassified tracks to determine appropriate adjustments.

## License
- This project is licensed under the MIT License. See the LICENSE file for details.



##### Repository Files
- Below is an overview of the key files included in the repository:
1. ntf_app.py: The main Python script containing the GUI application.
2. run_ntf.sh: Shell script to launch the application on Linux/macOS.
3. run_ntf.bat: Batch script to launch the application on Windows.
4. LICENSE: License file for the project.


## Detailed Explanation on How to Use the Script
- Setting Up Your Data - Input Directory Structure:
The input directory should contain subfolders for each track. Each subfolder should contain:
track.txt: A CSV file with X and Y columns representing the coordinates of the track.
track.avi: A video file of the track (required for advanced analysis).
Example Structure:

```
input_directory/
├── track_001/
│   ├── track.txt
│   └── track.avi
├── track_002/
│   ├── track.txt
│   └── track.avi
└── ...
```
- Running the Application - Launch the Application:
On Linux/macOS, run ./run_ntf.sh or double-click the desktop shortcut.
On Windows, double-click run_ntf.bat or its shortcut.

###  Configure the Application:
- Input Directory: Click Browse and select your input directory.
- Output Directory: Click Browse and select your desired output directory.
- Frame Rate (FPS): Enter the frame rate (e.g., 10).
- Processing Pipeline: Check or uncheck steps as needed.

### Set Clustering Options: 
- Clustering Method: Choose between SD Threshold and K-Means.SD Threshold Options: Enable Auto Threshold or input a custom threshold, or K-Means Options: Enable Auto-detect k or input the number of clusters.

###### Start Processing:
- Click Start Processing to begin the analysis.
- Monitor progress through the Progress Bar and Logs & Messages tab.

### View Results:
- After processing, interactive plots are displayed in the Viewing Panel.
- Navigate through plots using Previous Plot and Next Plot.

### Using Advanced Analysis
- Navigate to Advanced Analysis Tab, Click on the Advanced Analysis tab in the application, Adjust Thresholds:
- Use Auto Classification: Check to use default thresholds.
- Uncheck Use Auto Classification to input custom thresholds for:
1. Eccentricity
2. Solidity
3. Aspect Ratio
4. Circularity
5. Start Classification:

### Click Start Classification to classify tracks.
### Monitor progress and logs in the Advanced Analysis tab.
### Review Classification Results:

- Classified tracks are organized into worm, artifact, and unknown folders in the output directory.
- Analyze misclassified tracks and adjust thresholds if necessary.

### Tips for Effective Use
- Data Quality: Ensure that track.txt files contain accurate X and Y data.
Videos should be clear for effective feature extraction.
Threshold Adjustments: Start with auto classification and assess results.
Adjust thresholds incrementally based on the characteristics of misclassified tracks.
- Batch Processing: Organize datasets into subdirectories when using batch processing.
- Maintain consistent data structure across datasets.
- Monitoring Logs: Regularly check the Logs & Messages tab for errors or warnings.
Use the ntf_app.log file for detailed logging information.

### Troubleshooting
- Application Does Not Launch: Verify that the conda environment is correctly set up.
Ensure that the script (run_ntf.sh or run_ntf.bat) has the correct paths.
- Errors During Processing: Check the logs for specific error messages.
- Ensure that input files are in the correct format and directories.
- Incorrect Classification: Adjust classification thresholds in the Advanced Analysis tab. Review the shape features of misclassified tracks to inform threshold adjustments.

## Additional Resources
- Plotly Documentation: https://plotly.com/python/
- scikit-learn Documentation: https://scikit-learn.org/stable/
- OpenCV Documentation: https://docs.opencv.org/
- PyQt5 Documentation: https://www.riverbankcomputing.com/static/Docs/PyQt5/


