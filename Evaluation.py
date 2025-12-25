import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# EVALUATION
#===============================================================================
def evaluateModel(model, test_data): # Evaluate model simply on test data
    print("MODEL EVALUATION")
    print("="*60)
    results = model.evaluate(test_data) 
    print(f"Loss:     {results[0]:.4f}")
    print(f"Accuracy: {results[1]:.4f} ({results[1]*100:.2f}%)")
    print("="*60)
    return results

def evaluation(y_true, y_pred): # Evalaute model more detailed with confusion matrix and metrics
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("\n" + "="*60)
    print("DETAILED EVALUATION RESULTS")
    print("="*60)
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    print(f"                 Predicted Normal  |  Predicted Stunting")
    print(f"  Actual Normal:      {tn:>4}       |      {fp:>4}")
    print(f"  Actual Stunting:    {fn:>4}       |      {tp:>4}")
    print(f"\n  True Positive (TP):  {tp:>3} - Correctly detected stunting")
    print(f"  True Negative (TN):  {tn:>3} - Correctly detected normal")
    print(f"  False Positive (FP): {fp:>3} - Normal flagged as stunting")
    print(f"  False Negative (FN): {fn:>3} - Stunting cases missed ⚠️")
    
    # Key Metrics
    print(f"\nPerformance Metrics:")
    print(f"  Accuracy:    {accuracy:.4f} ({accuracy*100:>6.2f}%) - Overall correctness")
    print(f"  Precision:   {precision:.4f} ({precision*100:>6.2f}%) - Reliability when predicting stunting")
    print(f"  Recall:      {recall:.4f} ({recall*100:>6.2f}%) - Ability to detect all stunting cases")
    print(f"  F1-Score:    {f1:.4f} ({f1*100:>6.2f}%) - Balance between precision & recall")

    print(f"\nQuick Assessment:")
    if recall >= 0.85:
        print(f"Good detection rate - catching {recall*100:.1f}% of stunting cases")
    else:
        print(f"Detection rate is {recall*100:.1f}% - missing {fn} stunting cases")
    
    if precision >= 0.80:
        print(f"Good precision - only {fp} false alarms")
    else:
        print(f"{fp} false alarms - normal children flagged as stunting")
    
    print("="*60 + "\n")

    results = {
        'Confusion Matrix': cm,
        'True Negatives': tn,
        'False Positives': fp,
        'False Negatives': fn,
        'True Positives': tp,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }
    return results