import tensorflow as tf
from tensorflow import keras 

# MODEL 
#=============================================================================== DONE
def modelDefinition(config): # Definisi architect model CNN
    layers = keras.layers
    model = keras.Sequential([
        # total
        layers.Conv2D(32, (3, 3), activation='relu', input_shape = config['input_shape']), # ekstrak dari layer konvolusi 1
        layers.BatchNormalization(), # normalisasi batch
        layers.MaxPooling2D((2, 2)), # reduksi dimensi

        layers.Conv2D(64, (3, 3), activation='relu'), # layer 2
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)), 

        layers.Conv2D(128, (3, 3), activation='relu'), # layer 3
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(), # gepengin jadi vektor 1D x.y.z
        layers.Dense(128, activation='relu'), # fully connected layer tiap neuron
        layers.Dropout(0.3), # regularisasi
        layers.Dense(1, activation='sigmoid') # output probabilitas klasifikasi 0-1
    ])
    return model

def compileModel(model, config): # Compile model dengan loss dan optimizer
    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=config['learning_rate']), # learning rate 0.001
        loss = 'binary_crossentropy', # 2 kelas
        metrics = ['accuracy']) # eval
    return model


