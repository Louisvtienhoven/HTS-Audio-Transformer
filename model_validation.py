import os
import re
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Path to model output
FILE_PATH = "model_output.txt"  # Change to your path if needed

# Step 1: Parse model output
pattern = re.compile(r"(.+\.wav) => Predicted: (\d), Prob: ([\d.]+), Ground Truth: (.+)")
entries = []

with open(FILE_PATH, "r") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            filename, pred, prob, gt = match.groups()
            entries.append({
                "filename": filename,
                "predicted": int(pred),
                "prob": float(prob),
                "gt": gt.strip()
            })

# Normalize filenames (lowercase + basename)
def normalize(filename):
    return os.path.basename(filename).lower()

# Step 2: Build map from original to augmented
augmented_map = defaultdict(list)
for entry in entries:
    if "_augmented_" in entry["filename"].lower():
        base = normalize(entry["filename"].rsplit("_augmented_", 1)[0] + ".wav")
        augmented_map[base].append(entry)

# Step 3: Evaluate predictions
true_labels = []
predicted_labels = []

TP_files, FP_files, TN_files, FN_files = [], [], [], []
anomalous_unseen_files = []
anomalous_augmented_files = []

count_eval = 0
count_unseen = 0
count_anomalous = 0

for entry in entries:
    raw_filename = entry["filename"]
    filename = normalize(raw_filename)
    is_augmented = "_augmented_" in filename
    is_unseen = entry["gt"].upper() == "N/A"
    pred = entry["predicted"]

    # --- Unseen file ---
    if is_unseen:
        count_unseen += 1
        is_anomalous = filename in augmented_map
        label = 1 if is_anomalous else 0

        # Defensive override: if file looks augmented, force label = 1
        if "_augmented_" in filename:
            label = 1

        if label == 1:
            anomalous_unseen_files.append(raw_filename)

        true_labels.append(label)
        predicted_labels.append(pred)
        count_eval += 1
        if label == 1:
            count_anomalous += 1

        # Confusion matrix assignment
        if label == 1 and pred == 1:
            TP_files.append(raw_filename)
        elif label == 1 and pred == 0:
            FN_files.append(raw_filename)
        elif label == 0 and pred == 0:
            TN_files.append(raw_filename)
        elif label == 0 and pred == 1:
            # Sanity check
            if "_augmented_" in raw_filename.lower():
                print(f"❌ WARNING: Augmented file misclassified as FP: {raw_filename}")
            FP_files.append(raw_filename)

    # --- Augmented file (GT = 1 by definition) ---
    # --- Augmented file used only if GT is N/A (i.e. unseen augmentation) ---
    elif is_augmented and entry["gt"].upper() == "N/A":

        label = 1
        true_labels.append(label)
        predicted_labels.append(pred)
        anomalous_augmented_files.append(raw_filename)
        count_eval += 1
        count_anomalous += 1

        if pred == 1:
            TP_files.append(raw_filename)
        else:
            FN_files.append(raw_filename)

# Step 4: Report
print(f"\n✅ Total evaluated files: {count_eval}")
print(f"📁 Unseen files: {count_unseen}")
print(f"📎 Total anomalous: {count_anomalous}")
print(f"🧪 Anomalous unseen: {len(anomalous_unseen_files)}")
print(f"🧪 Anomalous augmented: {len(anomalous_augmented_files)}")

# Metrics
if true_labels:
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, zero_division=0)
    recall = recall_score(true_labels, predicted_labels, zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, zero_division=0)

    print("\n📊 Evaluation Metrics:")
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")
else:
    print("⚠️ No valid files found for evaluation.")

# Confusion matrix summary
print("\n🔍 Prediction Breakdown:")
print(f"TP (True Positives): {len(TP_files)}")
print(f"FP (False Positives): {len(FP_files)}")
print(f"TN (True Negatives): {len(TN_files)}")
print(f"FN (False Negatives): {len(FN_files)}")

# Print file lists
def print_file_list(label, file_list):
    print(f"\n📂 {label} ({len(file_list)} files):")
    for f in file_list:
        print(f" - {f}")

print_file_list("True Positives (TP)", TP_files)
print_file_list("False Positives (FP)", FP_files)
print_file_list("True Negatives (TN)", TN_files)
print_file_list("False Negatives (FN)", FN_files)
