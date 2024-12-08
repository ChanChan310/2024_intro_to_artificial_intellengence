import os
import csv

def update_meta_csv(meta_csv_path, results_dir):
    # Define the label for the new data
    new_label = "1"

    # List to store new entries
    new_entries = []

    # Iterate through all subdirectories under the results directory
    for subdir, _, files in os.walk(results_dir):
        for file in files:
            if file.endswith("_flowavenet_audio.wav"):
                # Construct the relative path for the file
                relative_path = os.path.join(os.path.relpath(subdir, start='.'), file).replace("\\", "/")
                # Create a new entry
                new_entries.append([relative_path, new_label])

    # Open the meta.csv file in append mode and write new entries
    with open(meta_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for entry in new_entries:
            writer.writerow(entry)

    print(f"Added {len(new_entries)} new entries to {meta_csv_path}.")

# Paths to the meta.csv file and results directory
meta_csv_path = "train_dataset/meta.csv"  # Replace with the actual path to your meta.csv
results_dir = "train_dataset/generate_audio/"  # Replace with the actual path to your results directory

# Call the function
update_meta_csv(meta_csv_path, results_dir)
