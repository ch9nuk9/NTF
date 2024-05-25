import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from scipy.stats import gaussian_kde, kurtosis, skew, entropy
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from statsmodels.regression.mixed_linear_model import MixedLM
import statsmodels.api as sm
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import ttest_rel
import datetime
import threading

def load_folder():
    try:
        folder_path = filedialog.askdirectory()
        if folder_path:
            process_folder_threaded(folder_path)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load folder: {e}")

def process_folder_threaded(folder_path):
    try:
        thread = threading.Thread(target=process_folder, args=(folder_path,))
        thread.start()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to start processing thread: {e}")

def create_output_directory(base_path):
    try:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(base_path, f'output_{timestamp}')
        os.makedirs(output_dir, exist_ok=True)

        filtered_dir_kde = os.path.join(output_dir, 'filtered_kde')
        filtered_dir_lme = os.path.join(output_dir, 'filtered_lme')
        plots_dir = os.path.join(output_dir, 'plots')
        stats_dir = os.path.join(output_dir, 'stats')

        os.makedirs(filtered_dir_kde, exist_ok=True)
        os.makedirs(filtered_dir_lme, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        os.makedirs(stats_dir, exist_ok=True)

        return output_dir, filtered_dir_kde, filtered_dir_lme, plots_dir, stats_dir
    except Exception as e:
        messagebox.showerror("Error", f"Failed to create output directory: {e}")
        return None, None, None, None, None

def process_folder(folder_path):
    output_dir, filtered_dir_kde, filtered_dir_lme, plots_dir, stats_dir = create_output_directory(folder_path)
    if not output_dir:
        return

    all_filtered_data_kde = []
    all_filtered_data_lme = []
    stationary_elements_kde = []
    nonstationary_elements_kde = []
    stationary_elements_lme = []
    nonstationary_elements_lme = []

    subfolders = [subfolder for subfolder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, subfolder))]
    total_subfolders = len(subfolders)
    valid_subfolder_count = 0
    skipped_subfolder_count = 0
    skipped_folders = []
    stationary_frames_record = []

    for idx, subfolder in enumerate(subfolders):
        subfolder_path = os.path.join(folder_path, subfolder)
        txt_files = [f for f in os.listdir(subfolder_path) if f.endswith('.txt')]
        if txt_files:
            txt_file_path = os.path.join(subfolder_path, txt_files[0])
            print(f"Processing file: {txt_file_path}")
            try:
                data = pd.read_csv(txt_file_path, sep=r'\s+', names=['frame', 'time', 'X', 'Y'])
                data['X'] = pd.to_numeric(data['X'], errors='coerce')
                data['Y'] = pd.to_numeric(data['Y'], errors='coerce')
                data['time'] = pd.to_numeric(data['time'], errors='coerce')
                data.dropna(subset=['X', 'Y', 'time'], inplace=True)
            except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
                print(f"Error reading file {txt_file_path}: {e}")
                skipped_subfolder_count += 1
                skipped_folders.append(subfolder)
                update_progress(idx, total_subfolders)
                continue

            data['subject_id'] = subfolder
            data['dX'] = data['X'].diff().fillna(0)
            data['dY'] = data['Y'].diff().fillna(0)
            data['distance'] = np.sqrt(data['dX']**2 + data['dY']**2)

            try:
                data = calculate_parameters(data)
            except Exception as e:
                print(f"Error calculating parameters for subject {subfolder}: {e}")
                skipped_subfolder_count += 1
                skipped_folders.append(subfolder)
                update_progress(idx, total_subfolders)
                continue

            if data['distance'].std() == 0:
                print(f"Subject {subfolder} has no movement variability. Skipping.")
                skipped_subfolder_count += 1
                skipped_folders.append(subfolder)
                update_progress(idx, total_subfolders)
                continue

            try:
                kde_threshold, kurt, skw = find_optimal_threshold(data)
                lme_threshold, random_intercept_std = calculate_lme_threshold(data)
            except Exception as e:
                print(f"Error finding optimal threshold for subject {subfolder}: {e}")
                skipped_subfolder_count += 1
                skipped_folders.append(subfolder)
                update_progress(idx, total_subfolders)
                continue

            print(f"Subject {subfolder} - KDE Threshold: {kde_threshold}")
            print(f"Subject {subfolder} - LME Threshold: {lme_threshold}")

            data["true_label"] = (data['distance'] > kde_threshold).astype(int)

            try:
                filtered_data_kde, stationary_ratio_kde = advanced_filtering(data, kde_threshold)
                print(f"Subject {subfolder} - KDE Stationary Ratio after filtering: {stationary_ratio_kde:.2f}")
            except Exception as e:
                print(f"Error during advanced filtering for subject {subfolder}: {e}")
                skipped_subfolder_count += 1
                skipped_folders.append(subfolder)
                update_progress(idx, total_subfolders)
                continue
            
            if stationary_ratio_kde <= 0.95:
                all_filtered_data_kde.append(filtered_data_kde)
                nonstationary_elements_kde.append(subfolder)
                valid_subfolder_count += 1
            else:
                stationary_elements_kde.append(subfolder)

            stationary_lme = data['distance'] < lme_threshold
            stationary_ratio_lme = stationary_lme.mean()
            print(f"Subject {subfolder} - LME Stationary Ratio: {stationary_ratio_lme:.2f}")

            if stationary_ratio_lme <= 0.95:
                filtered_data_lme = data[~stationary_lme]
                all_filtered_data_lme.append(filtered_data_lme)
                nonstationary_elements_lme.append(subfolder)
                valid_subfolder_count += 1
            else:
                stationary_elements_lme.append(subfolder)

            stationary_frames_record.append((stationary_ratio_kde, stationary_ratio_lme))

        update_progress(idx, total_subfolders)

    if all_filtered_data_kde:
        combined_filtered_data_kde = pd.concat(all_filtered_data_kde)
    else:
        combined_filtered_data_kde = pd.DataFrame()
    
    if all_filtered_data_lme:
        combined_filtered_data_lme = pd.concat(all_filtered_data_lme)
    else:
        combined_filtered_data_lme = pd.DataFrame()

    if not combined_filtered_data_kde.empty:
        save_filtered_data(combined_filtered_data_kde, filtered_dir_kde, "KDE")
    if not combined_filtered_data_lme.empty:
        save_filtered_data(combined_filtered_data_lme, filtered_dir_lme, "LME")

    save_indexed_elements(stationary_elements_kde, os.path.join(filtered_dir_kde, 'stationary_elements.txt'))
    save_indexed_elements(nonstationary_elements_kde, os.path.join(filtered_dir_kde, 'nonstationary_elements.txt'))
    save_indexed_elements(stationary_elements_lme, os.path.join(filtered_dir_lme, 'stationary_elements.txt'))
    save_indexed_elements(nonstationary_elements_lme, os.path.join(filtered_dir_lme, 'nonstationary_elements.txt'))

    progress_bar['value'] = 100
    progress_label['text'] = 'Processing Complete'
    valid_label['text'] = f'Valid Subfolders: {valid_subfolder_count}'
    skipped_label['text'] = f'Skipped Subfolders: {skipped_subfolder_count}'

    if not combined_filtered_data_kde.empty and not combined_filtered_data_lme.empty:
        kde_metrics, lme_metrics = cross_validate(pd.concat([combined_filtered_data_kde, combined_filtered_data_lme]))
        plot_performance_comparison(kde_metrics, lme_metrics, plots_dir)
        perform_statistical_tests(kde_metrics, lme_metrics, stats_dir)
    
        subject_ids = combined_filtered_data_kde['subject_id'].unique()
        similarities, filtered_subjects = compare_pdfs(combined_filtered_data_kde, subject_ids)
    
        save_pdf_comparison_results(similarities, filtered_subjects, stats_dir)
        plot_pdf_comparison(combined_filtered_data_kde, subject_ids, plots_dir)

def update_progress(current, total):
    progress = (current + 1) / total * 100
    progress_bar['value'] = progress
    progress_label['text'] = f'Processing {current + 1} of {total} subfolders'
    root.update_idletasks()

def find_optimal_threshold(data, manual_threshold=None):
    distances = data['distance'].values
    
    try:
        density = gaussian_kde(distances)
    except np.linalg.LinAlgError:
        pca = PCA(n_components=1)
        distances = pca.fit_transform(distances.reshape(-1, 1)).flatten()
        density = gaussian_kde(distances)
    
    xs = np.linspace(0, np.max(distances), 1000)
    density_values = density(xs)
    
    if manual_threshold is None:
        threshold = xs[np.argmax(density_values > density_values.max() * 0.05)]
    else:
        threshold = np.percentile(distances, manual_threshold)
    
    kurt = kurtosis(distances)
    skw = skew(distances)
    
    if threading.current_thread() is threading.main_thread():
        plt.figure(figsize=(10, 6))
        plt.plot(xs, density_values, label='Density')
        plt.axvline(threshold, color='r', linestyle='--', label=f'Optimal Threshold: {threshold:.2f}')
        plt.xlabel('Distance')
        plt.ylabel('Density')
        plt.title('Optimal Threshold Determination')
        plt.legend()
        plt.text(0.95, 0.95, f'Kurtosis: {kurt:.2f}\nSkewness: {skw:.2f}', 
                 transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right')
        plt.show()
    
    return threshold, kurt, skw

def calculate_lme_threshold(data):
    try:
        model = MixedLM.from_formula("distance ~ 1", groups=data["subject_id"], data=data)
        result = model.fit()
        fixed_intercept = result.params["Intercept"]
        random_intercept_std = np.std(list(result.random_effects.values()))
        return fixed_intercept, random_intercept_std
    except Exception as e:
        raise ValueError(f"Error in LME threshold calculation: {e}")

def calculate_parameters(data):
    try:
        data['velocity'] = np.sqrt(data['dX']**2 + data['dY']**2) / data['time'].diff().fillna(1)
        data['speed'] = data['velocity'].rolling(window=3, min_periods=1).mean()
        data['turn_angle'] = np.arctan2(data['dY'], data['dX']).diff().fillna(0).abs()
        return data
    except Exception as e:
        raise ValueError(f"Error calculating parameters: {e}")

def advanced_filtering(data, kde_threshold, total_distance_threshold=10):
    try:
        data['total_distance'] = data['distance'].cumsum()
        filtered_data = data[data['total_distance'] > total_distance_threshold]
        stationary_kde = filtered_data['distance'] < kde_threshold
        stationary_ratio_kde = stationary_kde.mean()

        if stationary_ratio_kde <= 0.95:
            return filtered_data, stationary_ratio_kde
        else:
            return pd.DataFrame(), 1.0
    except Exception as e:
        raise ValueError(f"Error during advanced filtering: {e}")

def analyze_movement_patterns(data):
    try:
        kurt = kurtosis(data['distance'])
        skw = skew(data['distance'])
        return kurt, skw
    except Exception as e:
        raise ValueError(f"Error analyzing movement patterns: {e}")

def save_filtered_data(data, directory, method):
    try:
        save_path = os.path.join(directory, f'filtered_data_{method}.csv')
        data[['subject_id', 'frame', 'time', 'X', 'Y', 'velocity', 'speed', 'turn_angle']].to_csv(save_path, sep=' ', index=False)
        messagebox.showinfo("Success", f"File has been saved successfully in {save_path}.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save the file: {e}")

def save_indexed_elements(elements, filepath):
    try:
        with open(filepath, 'w') as f:
            for element in elements:
                f.write(f"{element}\n")
    except Exception as e:
        raise ValueError(f"Error saving indexed elements: {e}")

def calculate_metrics(true_labels, predicted_labels):
    try:
        tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, zero_division=0)
        recall = recall_score(true_labels, predicted_labels, zero_division=0)
        f1 = f1_score(true_labels, predicted_labels, zero_division=0)
        return tpr, fpr, accuracy, precision, recall, f1
    except Exception as e:
        raise ValueError(f"Error calculating metrics: {e}")

def cross_validate(data, k=5):
    kde_metrics = []
    lme_metrics = []
    try:
        kf = KFold(n_splits=k)
        for train_index, test_index in kf.split(data):
            train_data, test_data = data.iloc[train_index], data.iloc[test_index]

            kde_threshold, _, _ = find_optimal_threshold(train_data)
            test_data.loc[:, "kde_pred"] = test_data["distance"] < kde_threshold
            kde_metrics.append(calculate_metrics(test_data["true_label"], test_data["kde_pred"]))

            lme_threshold, _ = calculate_lme_threshold(train_data)
            test_data.loc[:, "lme_pred"] = test_data["distance"] < lme_threshold
            lme_metrics.append(calculate_metrics(test_data["true_label"], test_data["lme_pred"]))

        return kde_metrics, lme_metrics
    except Exception as e:
        raise ValueError(f"Error during cross-validation: {e}")

def perform_statistical_tests(kde_metrics, lme_metrics, stats_dir):
    try:
        kde_acc = [m[2] for m in kde_metrics]
        lme_acc = [m[2] for m in lme_metrics]

        if kde_acc and lme_acc:
            t_stat, p_value = ttest_rel(kde_acc, lme_acc)
            print(f"Paired t-test: t-statistic = {t_stat}, p-value = {p_value}")

            with open(os.path.join(stats_dir, 'paired_t_test_results.txt'), 'w') as f:
                f.write(f"Paired t-test: t-statistic = {t_stat}, p-value = {p_value}\n")

            kde_preds = np.concatenate([m[1] for m in kde_metrics])
            lme_preds = np.concatenate([m[1] for m in lme_metrics])
            true_labels = np.concatenate([m[0] for m in kde_metrics])

            contingency_table = [[sum((kde_preds == true_labels) & (lme_preds == true_labels)),
                                sum((kde_preds == true_labels) & (lme_preds != true_labels))],
                                [sum((kde_preds != true_labels) & (lme_preds == true_labels)),
                                sum((kde_preds != true_labels) & (lme_preds != true_labels))]]

            mcnemar_result = mcnemar(contingency_table)
            print(f"McNemar's test: statistic = {mcnemar_result.statistic}, p-value = {mcnemar_result.pvalue}")

            with open(os.path.join(stats_dir, 'mcnemar_test_results.txt'), 'w') as f:
                f.write(f"McNemar's test: statistic = {mcnemar_result.statistic}, p-value = {mcnemar_result.pvalue}\n")
        else:
            raise ValueError("Need at least one array to concatenate for statistical tests.")
    except Exception as e:
        raise ValueError(f"Error performing statistical tests: {e}")

def plot_performance_comparison(kde_metrics, lme_metrics, plots_dir):
    try:
        metrics = ["TPR", "FPR", "Accuracy", "Precision", "Recall", "F1 Score"]
        kde_avg_metrics = np.mean(kde_metrics, axis=0)
        lme_avg_metrics = np.mean(lme_metrics, axis=0)

        x = np.arange(len(metrics))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width/2, kde_avg_metrics, width, label='KDE')
        ax.bar(x + width/2, lme_avg_metrics, width, label='LME')

        ax.set_xlabel('Metrics')
        ax.set_ylabel('Scores')
        ax.set_title('Performance Metrics Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()

        plot_path = os.path.join(plots_dir, 'performance_comparison.png')
        plt.savefig(plot_path)
        plt.show()
    except Exception as e:
        raise ValueError(f"Error plotting performance comparison: {e}")

def visualize_stationary_vs_moving(stationary_ratios, plots_dir):
    try:
        kde_ratios = [r[0] for r in stationary_ratios]
        lme_ratios = [r[1] for r in stationary_ratios]

        plt.figure(figsize=(10, 6))
        plt.hist(kde_ratios, bins=20, alpha=0.7, label='KDE Stationary Ratios')
        plt.hist(lme_ratios, bins=20, alpha=0.7, label='LME Stationary Ratios')
        plt.axvline(0.95, color='r', linestyle='--', label='Threshold for Skipping')
        plt.xlabel('Stationary Ratio')
        plt.ylabel('Count')
        plt.title('Distribution of Stationary vs. Moving Elements')
        plt.legend()

        plot_path = os.path.join(plots_dir, 'stationary_vs_moving_distribution.png')
        plt.savefig(plot_path)
        plt.show()
    except Exception as e:
        raise ValueError(f"Error visualizing stationary vs. moving: {e}")

def plot_pdf_comparison(data, subject_ids, plots_dir):
    try:
        plt.figure(figsize=(12, 6))
        for subject_id in subject_ids:
            subject_data = data.loc[data['subject_id'] == subject_id]
            distances = subject_data['distance'].values
            density = gaussian_kde(distances)
            xs = np.linspace(0, np.max(distances), 1000)
            density_values = density(xs)
            plt.plot(xs, density_values, label=f'Subject {subject_id}')
        plt.xlabel('Distance')
        plt.ylabel('Density')
        plt.title('PDF Comparison')
        plt.legend()

        plot_path = os.path.join(plots_dir, 'pdf_comparison.png')
        plt.savefig(plot_path)
        plt.show()
    except Exception as e:
        raise ValueError(f"Error plotting PDF comparison: {e}")

def calculate_kl_divergence(pdf1, pdf2):
    try:
        return entropy(pdf1, pdf2)
    except Exception as e:
        raise ValueError(f"Error calculating KL divergence: {e}")

def compare_pdfs(data, subject_ids, threshold=0.1):
    try:
        similarities = []
        for i, subject_id1 in enumerate(subject_ids):
            for j, subject_id2 in enumerate(subject_ids):
                if i >= j:
                    continue
                data1 = data.loc[data['subject_id'] == subject_id1]['distance'].values
                data2 = data.loc[data['subject_id'] == subject_id2]['distance'].values

                density1 = gaussian_kde(data1)(np.linspace(0, np.max(data1), 1000))
                density2 = gaussian_kde(data2)(np.linspace(0, np.max(data2), 1000))

                similarity = calculate_kl_divergence(density1, density2)
                similarities.append((subject_id1, subject_id2, similarity))

        filtered_subjects = [s for s in similarities if s[2] < threshold]
        return similarities, filtered_subjects
    except Exception as e:
        raise ValueError(f"Error comparing PDFs: {e}")

def save_pdf_comparison_results(similarities, filtered_subjects, stats_dir):
    try:
        with open(os.path.join(stats_dir, 'pdf_comparison_results.txt'), 'w') as f:
            for subject_id1, subject_id2, similarity in similarities:
                f.write(f"Subjects {subject_id1} and {subject_id2} have KL divergence of {similarity:.4f}\n")

        with open(os.path.join(stats_dir, 'filtered_subjects.txt'), 'w') as f:
            for subject_id1, subject_id2, similarity in filtered_subjects:
                f.write(f"Filtered Subjects {subject_id1} and {subject_id2} with KL divergence of {similarity:.4f}\n")
    except Exception as e:
        raise ValueError(f"Error saving PDF comparison results: {e}")

def set_manual_threshold():
    try:
        manual_threshold = float(manual_threshold_entry.get())
        if 0 <= manual_threshold <= 100:
            find_optimal_threshold(data=None, manual_threshold=manual_threshold)
        else:
            messagebox.showerror("Error", "Threshold percentage must be between 0 and 100.")
    except ValueError:
        messagebox.showerror("Error", "Invalid input. Please enter a numerical value.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to set manual threshold: {e}")

root = tk.Tk()
root.title("Nematode Tracking Filter")

tab_control = ttk.Notebook(root)

tab1 = ttk.Frame(tab_control)
tab2 = ttk.Frame(tab_control)
tab3 = ttk.Frame(tab_control)

tab_control.add(tab1, text='Main')
tab_control.add(tab2, text='Statistics')
tab_control.add(tab3, text='Visualization')

tab_control.pack(expand=1, fill='both')

# Main Tab
load_button = tk.Button(tab1, text="Load Folder", command=load_folder)
load_button.pack(pady=10)

manual_threshold_label = tk.Label(tab1, text="Manual Threshold Percentage:")
manual_threshold_label.pack()
manual_threshold_entry = ttk.Entry(tab1)
manual_threshold_entry.pack()
manual_threshold_entry.insert(0, "5")
manual_threshold_button = tk.Button(tab1, text="Set Manual Threshold", command=set_manual_threshold)
manual_threshold_button.pack()

progress_frame = tk.Frame(tab1)
progress_frame.pack(pady=10)
progress_label = tk.Label(progress_frame, text="")
progress_label.pack()
progress_bar = ttk.Progressbar(progress_frame, orient="horizontal", length=250, mode="determinate")
progress_bar.pack()

count_frame = tk.Frame(tab1)
count_frame.pack(pady=10)
valid_label = tk.Label(count_frame, text="Valid Subfolders: 0")
valid_label.pack()
skipped_label = tk.Label(count_frame, text="Skipped Subfolders: 0")
skipped_label.pack()

# Statistics Tab
statistics_button = tk.Button(tab2, text="Calculate Statistics", command=lambda: perform_statistical_tests([], [], ""))
statistics_button.pack(pady=10)

# Visualization Tab
plot_comparison_button = tk.Button(tab3, text="Plot Performance Comparison", command=lambda: plot_performance_comparison([], [], ""))
plot_comparison_button.pack(pady=10)

plot_pdf_button = tk.Button(tab3, text="Plot PDF Comparison", command=lambda: plot_pdf_comparison(pd.DataFrame(columns=['subject_id', 'distance']), ['subject1', 'subject2'], ""))
plot_pdf_button.pack(pady=10)

root.mainloop()
