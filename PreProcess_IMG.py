import os
from PIL import Image
from ConfigHyperParameter import config
# Preprocess all images (resize and cleaning)
#===============================================================================
def preprocessAllIMG(dataset_dir="dataimg", output_dir="dataimg_processed", target_size=None):
    if target_size is None:
        target_size = config['target_size']
    total = 0
    removed = 0
    for root, dirs, files in os.walk(dataset_dir): # read all files in dataset_dir
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                total += 1
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        img = img.convert('L')
                        img = img.resize(target_size)
                        img.save(file_path)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    os.remove(file_path)
                    removed += 1

    print("Image preprocessing completed.")
    print(f"Total images processed: {total}")
    print(f"Total images removed due to errors: {removed}")
