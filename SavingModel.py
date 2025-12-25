import tensorflow as tf
from tensorflow import keras
import os

# SAVING MODEL
#===============================================================================
def saveModel(model, fileName = 'stunting_model.keras'): # Save dengan nama stunding_model.keras
    if not (fileName.endswith('.keras') or fileName.endswith('.h5')):
        fileName += '.keras'
    model.save(fileName)

def loadModel(fileName = 'stunting_model.keras'): # Jaga2 kalo kepake buat load stunting_model.keras
    if not os.path.exists(fileName):
        raise FileNotFoundError(f"The model {fileName} does not exist my man")
    model = keras.models.load_model(fileName)
    return model
