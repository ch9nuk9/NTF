import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
from scipy.stats import gaussian_kde

def load_folder():
    # Opens a folder dialog to select the main folder containing subfolders
    folder_path = filedialog.askdirectory()
    if folder_path:
        process_folder(folder_path)

def process_folder(folder_path):
    # List to store filtered data for all subjects
    all_filtered_data = []
    
    # Traverse each subfolder
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        
        # Check if the subfolder path is indeed a directory
        if os.path.isdir(subfolder_path):
            # Find the .txt file in the subfolder
            txt_files = [f for f in os.listdir(subfolder_path) if f.endswith('.txt')]
            
            if txt_files:
                txt_file_path = os.path.join(subfolder_path, txt_files[0])
                print(f"Processing file: {txt_file_path}")
                
                # Read and process the data file
                data = pd.read_csv(txt_file_path, delim_whitespace=True, names=['frame', 'time', 'X', 'Y'])
                data['subject_id'] = subfolder  # Add the subfolder name as the subject ID
                
                # Calculate differences and distances
                data['dX'] = data['X'].diff().fillna(0)
                data['dY'] = data['Y'].diff().fillna(0)
                data['distance'] = np.sqrt(data['dX']**2 + data['dY']**2)
                
                # Find the optimal threshold to determine stationary frames
                optimal_threshold = find_optimal_threshold(data)
                print(f"Subject {subfolder} - Optimal Threshold: {optimal_threshold}")
                
                # Determine if the subject is predominantly stationary
                stationary = data['distance'] < optimal_threshold
                stationary_ratio = stationary.mean()
                print(f"Subject {subfolder} - Stationary Ratio: {stationary_ratio:.2f}")
                
                # If the subject is predominantly stationary (e.g., more than 95% stationary), skip it
                if stationary_ratio > 0.95:
                    print(f"Subject {subfolder} is predominantly stationary. Skipping.")
                    continue
                
                # Determine stationary frames based on the optimal threshold
                stationary_frames = data.groupby('frame').apply(lambda group: (group['distance'] < optimal_threshold).all())
                filtered_data = data[~data['frame'].isin(stationary_frames[stationary_frames].index)]
                
                # Append the filtered data for this subject to the list
                all_filtered_data.append(filtered_data)
    
    # Concatenate all filtered data into a single DataFrame
    combined_filtered_data = pd.concat(all_filtered_data)
    
    # Save the filtered data to a new file
    save_path = filedialog.asksaveasfilename(defaultextension=".txt")
    if save_path:
        combined_filtered_data[['subject_id', 'frame', 'time', 'X', 'Y']].to_csv(save_path, sep=' ', index=False)
        messagebox.showinfo("Success", "File has been saved successfully.")

def find_optimal_threshold(data):
    # Use Gaussian KDE to find the optimal threshold for distances
    distances = data['distance'].values
    density = gaussian_kde(distances)
    xs = np.linspace(0, np.max(distances), 1000)
    density_values = density(xs)
    
    # Determine the threshold where the density is 5% of the maximum density
    threshold = xs[np.argmax(density_values > density_values.max() * 0.05)]
    
    # Plot the density and threshold
    plt.figure(figsize=(10, 6))
    plt.plot(xs, density_values, label='Density')
    plt.axvline(threshold, color='r', linestyle='--', label=f'Optimal Threshold: {threshold:.2f}')
    plt.xlabel('Distance')
    plt.ylabel('Density')
    plt.title('Optimal Threshold Determination')
    plt.legend()
    plt.show()
    
    return threshold

# Create the main window
root = tk.Tk()
root.title("Nematode Tracking Filter")

# Create a button to load the folder
load_button = tk.Button(root, text="Load Folder", command=load_folder)
load_button.pack(pady=20)

# Run the application
root.mainloop()
