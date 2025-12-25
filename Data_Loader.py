from tensorflow import keras
ImageDataGenerator = keras.preprocessing.image.ImageDataGenerator
# DATA LOADER
#===============================================================================
# load dataset dengan perubahan variasi untuk training data
def load_dataset(train_dir, val_dir, config):
    train_datagen = ImageDataGenerator( # augmnetasi data untuk variasi data training
        rescale=1./255,
        rotation_range=20,
        brightness_range=[0.8,1.2],
        width_shift_range=0.2, 
        height_shift_range=0.2,
        shear_range=0.2, 
        zoom_range=0.2, 
        horizontal_flip=True, 
        fill_mode='nearest' 
    )
    train_data = train_datagen.flow_from_directory( # data training dengan augmentasi
        train_dir,
        target_size=config['target_size'],
        batch_size=config['batch_size'],
        class_mode='binary',
        color_mode='grayscale'
    )
    val_datagen = ImageDataGenerator(rescale=1./255) # gosah augmentasi
    val_data = val_datagen.flow_from_directory( # data validation tanpa augmentasi
        val_dir,
        target_size=config['target_size'],
        batch_size=config['batch_size'],
        class_mode='binary',
        color_mode='grayscale'
    )
    return train_data, val_data
 
def test_dataset(test_dir, config): # test dataset tanpa augmentasi
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_data = test_datagen.flow_from_directory(
        test_dir,
        target_size=config['target_size'],
        batch_size=config['batch_size'],
        class_mode='binary',
        shuffle = False,
        color_mode='grayscale'
    )
    return test_data