# NTF: Nematode Tracking Data Filter and Threshold Refining Tool
## Overview
This repository contains Python-based tools for processing nematode tracking data. It includes an enhanced script for filtering tracking data to remove stationary subjects, refining movement thresholds using Gaussian Kernel Density Estimation (KDE), and instructions for creating a standalone executable (.exe) for easy distribution.

## Features
- **Data Filtering**: Process tracking data to filter out predominantly stationary subjects.
- **Threshold Refining**: Utilize Gaussian KDE to determine optimal movement thresholds.
- **Standalone .exe Creation**: Instructions to package the Python script into a standalone executable for Windows.

## Prerequisites
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
    conda install pandas numpy matplotlib scipy tk
    ```

3. **Clone the Repository**:
    ```bash
    git clone https://github.com/ch9nuk9/NTF.git
    cd (path/to/repo)
    ```

### OR

1. **Simply run the setup_environment.yaml**:
    ```bash
    conda env create -f setup_environment.yaml
    conda activate NTF

### OR

1. **Simply run the setup_environment.sh (Creates the conda environment NTF, activates and installs all the dependencies within the NTF)**:
    ```bash
    bash setup_environment.sh
###
###
###
### Running the Enhanced Script

1. **Activate the Environment**:
    ```bash
    conda activate NTF
    ```

2. **Run the Script**:
    ```bash
    python NTF_script.py
    ```

## Creating a Standalone Executable (Only if necessary)

1. **Install PyInstaller**:
    ```bash
    conda install pyinstaller
    ```

2. **Generate the Executable**:
    ```bash
    pyinstaller --onefile NTF_script.py
    ```

3. **Locate the Executable**:
   The executable will be located in the `dist` directory.

### Optional: Creating an Installer

To create an installer for the .exe file, you can use [Inno Setup](http://www.jrsoftware.org/isinfo.php). Follow the Inno Setup wizard to package your executable into an installer.

## Usage
### Enhanced Script

1. **Load Folder**: 
    - Click the "Load Folder" button to select the main folder containing subfolders with tracking data.
    - Each subfolder should contain a `.txt` file with tracking data in four columns: `frame`, `time`, `X`, `Y`.

2. **Process Data**:
    - The script will process each subfolder's data, filter out predominantly stationary subjects, and save the filtered data to a new file.
    
      The find_optimal_threshold Function
    - Kernel Density Estimation (KDE): The core idea is to estimate the probability density function (PDF) of the distances between consecutive frames.
      KDE is a non-parametric way to smooth out the distribution of data points, creating a curve that represents how likely different distances are to occur.
      In this code, it uses gaussian_kde from the SciPy library to perform KDE. This means it's assuming the underlying distribution of distances somewhat resembles a Gaussian (normal) distribution.
      density(xs) calculates the density values for a range of distances (xs). However, in very rare cases the data for a particular subject lies in a lower-dimensional subspace.
      This leads to a singular covariance matrix which the gaussian_kde algorithm cannot handle. This usually happens when there isn't enough variability in the data.
      In order to bypass situations like this a bypass script has been introduced where the data is pre-checked for variability. If there isn't enough variability, such sets will be skipped.
      Alternatively, dimensionality of datasets will be reduced through PCA before using KDE. 
    - Finding the Threshold: The code aims to find a threshold distance that separates "stationary" frames from "moving" frames. It does this in one of two ways:
    - Automatic: It finds the distance where the density drops to 5% of its maximum value. This is like saying, "Find the point where the likelihood of observing this distance or greater becomes quite low."
      This low-density region is likely where stationary frames cluster. xs[np.argmax(density_values > density_values.max() * 0.05)] achieves this by finding the first point where the density exceeds the 5% threshold.
    - Manual (Optional): The user can provide a manual threshold percentage through the GUI. np.percentile(distances, manual_threshold) calculates the corresponding distance value from the data's percentiles.
      For example, a 5% manual threshold would find the distance value below which 5% of the data falls.
    - Plotting: The function plots the density curve. This curve helps you visualize how distances are distributed. It also marks the calculated/manual threshold as a vertical line on the plot.
      This gives a visual way to assess if the threshold seems reasonable given the data's distribution.

      Why This Approach ?
    - Adapts to Data: KDE doesn't assume a specific shape for the distribution of distances. It lets the data guide the shape of the curve. This is important because the distribution of distances in tracking data can vary depending on the experiment and behavior.
    - Flexibility: The option for a manual threshold provides control if you have domain knowledge or specific requirements about what constitutes "stationary."

3. **Save Filtered Data**:
    - Choose a location to save the filtered data file when prompted.

### Standalone Threshold Refining Script

The standalone threshold refining script (`Threshold_script.py`) focuses solely on determining the optimal movement threshold using Gaussian KDE.

1. **Run the Standalone Script**:
    ```bash
    python Threshold_script.py
    ```

2. **Select Data File**:
    - Use the file dialog to select a `.txt` file containing tracking data.

3. **View Threshold Plot**:
    - The script will display a plot showing the density distribution of distances and the optimal threshold.

4. **Save Threshold**:
    - The optimal threshold will be printed in the console and can be used for further analysis.



