from projectFunctions import formatDataset
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os  
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, DetCurveDisplay
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, RocCurveDisplay, DetCurveDisplay
)
from sklearn.metrics import roc_auc_score




# Define HOG block sizes (in cells, not pixels)
block_sizes = [(1,1), (2, 2), (3, 3), (4, 4), (5,5), (6,6), (7,7)]
results = []

# Start ROC plot
plt.figure(figsize=(10, 6))
plt.title("ROC Curve Comparison by Block Size")
# Set axis limits
plt.ylim(0.5, 1.05)
plt.xlim(-0.01, 0.4)
line_styles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5)), (0, (1, 1))]

for block_dim in block_sizes:
    
    # Load training and test sets
    _, train_features, train_labels = formatDataset(
        tarfilePath='./ExampleSets/Daimler/DaimlerTrain.tar.gz',
        deleteDir=True,
        randomSeed=4402,
        blockDimensions=block_dim
    )

    _, test_features, test_labels = formatDataset(
        tarfilePath='./ExampleSets/Daimler/DaimlerTest.tar.gz',
        deleteDir=True,
        randomSeed=4402,
        blockDimensions=block_dim
    )

    # Train model
    model = LinearSVC(random_state=42, dual=False)
    model.fit(train_features, train_labels)

    # Predict and evaluate
    predictions = model.predict(test_features)
    decision_scores = model.decision_function(test_features)
    fpr, tpr, _ = roc_curve(test_labels, decision_scores)
    roc_auc = roc_auc_score(test_labels, decision_scores)


    # Save metrics
    results.append({
        "block_size": f"{block_dim[0]}x{block_dim[1]}",
        "accuracy": accuracy_score(test_labels, predictions),
        "f1": f1_score(test_labels, predictions)
    })

    # Plot ROC curve
    style = line_styles[block_sizes.index(block_dim) % len(line_styles)]
    plt.plot(fpr, tpr, linestyle=style, label=f"{block_dim[0]}x{block_dim[1]} (AUC = {roc_auc:.2f})")


# Show ROC plot
plt.xlabel("False Positive Rate (Positive Label: True)")
plt.ylabel("True Positive Rate (Positive Label: True)")
plt.title("ROC curves")
plt.grid(True)
plt.legend(title="Block Size")
plt.tight_layout()
plt.show()

# Save metrics to CSV and print
df = pd.DataFrame(results)
df.to_csv("block_size_ablation.csv", index=False)

# Plot Accuracy and F1 comparison
plt.figure(figsize=(10, 6))
plt.plot(df["block_size"], df["accuracy"], label="Accuracy", marker='o')
plt.plot(df["block_size"], df["f1"], label="F1 Score", marker='s')
plt.title("Accuracy and F1 Score by Block Size")
plt.xlabel("Block Size")
plt.ylabel("Score")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Cleanup
if os.path.exists("block_size_ablation.csv"):
    os.remove("block_size_ablation.csv")