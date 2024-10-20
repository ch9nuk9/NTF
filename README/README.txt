# Nematode Tracks Filter (NTF)

A comprehensive tool for analyzing and filtering nematode movement tracks from microscopy data.

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
- [Future Work](#future-work)
- [How to Cite This Tool](#how-to-cite-this-tool)
- [Feedback and Support](#feedback-and-support)
- [Important Notes](#important-notes)
- [Final Remarks](#final-remarks)

---

## Introduction

The **Nematode Tracks Filter (NTF)** is a Python-based graphical user interface (GUI) application designed to process, analyze, and classify movement tracks of nematodes (worms) captured through microscopy. The application facilitates the filtering of movement data, clustering of tracks based on movement characteristics, and classification of tracks to distinguish worms from artifacts or unknown entities.

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

Open a terminal and clone the repository from GitHub:

```bash
git clone https://github.com/ch9nuk9/NTF
