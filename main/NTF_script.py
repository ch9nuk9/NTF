import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
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
                try:
                    data = pd.read_csv(txt_file_path, delim_whitespace=True, names=['frame', 'time', 'X', 'Y'])
                except FileNotFoundError:
                    print(f"File not found: {txt_file_path}")
                    continue
                except pd.errors.EmptyDataError:
                    print(f"File is empty: {txt_file_path}")
                    continue
                except pd.errors.ParserError:
                    print(f"Error parsing file: {txt_file_path}")
                    continue
                
                # Add the subfolder name as the subject ID
                data['subject_id'] = subfolder  
                
                # Calculate differences and distances
                data['dX'] = data['X'].diff().fillna(0) # Calculate X displacement
                data['dY'] = data['Y'].diff().fillna(0) # Calculate Y displacement
                data['distance'] = np.sqrt(data['dX']**2 + data['dY']**2) # Euclidean distance
                
                # Find the optimal threshold to determine stationary frames
                try:
                    optimal_threshold = find_optimal_threshold(data)
                except Exception as e:
                    print(f"Error finding optimal threshold for subject {subfolder}: {e}")
                    continue
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
    if all_filtered_data:
        combined_filtered_data = pd.concat(all_filtered_data)
        
        save_path = filedialog.asksaveasfilename(defaultextension=".txt")
        if save_path:
            try:
                combined_filtered_data[['subject_id', 'frame', 'time', 'X', 'Y']].to_csv(save_path, sep=' ', index=False)
                messagebox.showinfo("Success", "File has been saved successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save the file: {e}")
    else:
        messagebox.showwarning("No Data", "No valid data to save.")

def find_optimal_threshold(data, manual_threshold=None):
    # Use Gaussian KDE to find the optimal threshold for distances
    distances = data['distance'].values
    density = gaussian_kde(distances)
    xs = np.linspace(0, np.max(distances), 1000)
    density_values = density(xs)
    
    if manual_threshold is None:
        # Determine the threshold where the density is 5% of the maximum density
        threshold = xs[np.argmax(density_values > density_values.max() * 0.05)]
    else:
        # Use the manually specified threshold percentage
        threshold = np.percentile(distances, manual_threshold)
        
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
def set_manual_threshold():
    try:
        manual_threshold = float(manual_threshold_entry.get())
        if 0 <= manual_threshold <= 100:
            find_optimal_threshold(data=None, manual_threshold=manual_threshold)
        else:
            messagebox.showerror("Error", "Threshold percentage must be between 0 and 100.")
    except ValueError:
        messagebox.showerror("Error", "Invalid input. Please enter a numerical value.")

# Create the main window
root = tk.Tk()
root.title("Nematode Tracking Filter")

# Create a button to load the folder
load_button = tk.Button(root, text="Load Folder", command=load_folder)
load_button.pack(pady=10)

# Add a manual threshold entry widget
manual_threshold_label = tk.Label(root, text="Manual Threshold Percentage:")
manual_threshold_label.pack()
manual_threshold_entry = ttk.Entry(root)
manual_threshold_entry.pack()
manual_threshold_entry.insert(0, "5")  # Default value
manual_threshold_button = tk.Button(root, text="Set Manual Threshold", command=set_manual_threshold)
manual_threshold_button.pack()

# Add progress bar
progress_frame = tk.Frame(root)
progress_frame.pack(pady=10)
progress_label = tk.Label(progress_frame, text="")
progress_label.pack()
progress_bar = ttk.Progressbar(progress_frame, orient="horizontal", length=300, mode="determinate")
progress_bar.pack()

# Run the application
root.mainloop()
