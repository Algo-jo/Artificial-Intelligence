import tensorflow as tf
from ConfigHyperParameter import config
from PreProcess_IMG import preprocessAllIMG
from ModelArchitect import modelDefinition, compileModel
from ModelTraining import trainModel
from Plotting import plot_history
from Data_Loader import load_dataset, test_dataset
from SavingModel import saveModel
from Evaluation import evaluateModel, evaluation
import json
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator

# MAIN PROGRAM
#===============================================================================

# Preprocess
print("[1/7] preprocessing all files...")
preprocessAllIMG(dataset_dir="dataimg", target_size=config['target_size'])

# Load dataset
print("[2/7] loading dataset...")
train_dir = 'dataimg/train'
val_dir = 'dataimg/valid'
test_dir = 'dataimg/test'
train_data, val_data = load_dataset(train_dir, val_dir, config)
test_data = test_dataset(test_dir, config)
print(f"Train samples: {train_data.n}")
print(f"Validation samples: {val_data.n}")
print(f"Test samples: {test_data.n}")

# Define and compile model
print("[3/7] defining and compiling model...")
model = modelDefinition(config)
model = compileModel(model, config)

# Train model
print("[4/7] training model...")
history = trainModel(model, train_data, val_data, config)

# Evaluate model
print("[5/7] evaluating model...")
evaluateModel(model, test_data)
# Predict on test data for detailed evaluation
y_pred = model.predict(test_data)
y_pred = (y_pred > 0.5).astype("int32").flatten()   
y_true = test_data.classes
evaluation(y_true, y_pred)

# Saving model
print("[6/7] saving model...")
saveModel(model, 'stunting_model.keras')
print("stunting_model.keras saved successfully.")

# Plot training history
print("[7/7] plotting training history...")
plot_history(history, model, save_path = 'training_history.png', show = False)

# Saving class indices
with open("class_indices.json", "w") as f:
    json.dump(train_data.class_indices, f)
print("class_indices.json saved successfully.")
