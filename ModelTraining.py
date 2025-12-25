import tensorflow as tf
from tensorflow import keras

# TRAINING (Supervised) Prediksi y dari x (Klasifikasi)
#=============================================================================== DONE
callbackFunction = [ 
    keras.callbacks.EarlyStopping(patience = 5, restore_best_weights = True), # stop kalo model makin cacat
    keras.callbacks.ReduceLROnPlateau(factor = 0.5, patience = 3),            # decrease learning rate
    keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only = True)   # simpan model terbagus
]

def trainModel(model, train_data, val_data, config):
    history = model.fit(
        train_data, # latian
        validation_data = val_data, # validasi
        epochs = config['epochs'], # repeat n kali
        callbacks = callbackFunction 
    )
    return history # train, val accuracy and loss + new learning rate
                   # Output history (hasil training)