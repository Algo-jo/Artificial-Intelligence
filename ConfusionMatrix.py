import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Data raw
cm = np.array([
    [83, 12],  # Actual Normal
    [9, 85]    # Actual Stunting
])

# Labels
labels = ["Normal", "Stunting"]

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(cmap='Blues', values_format='d', ax=ax)
plt.title("Confusion Matrix")

# Save as PNG
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')

# Show (optional)
plt.show()

print("Saved as confusion_matrix.png")
