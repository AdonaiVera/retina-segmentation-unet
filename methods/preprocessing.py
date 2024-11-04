import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocess_data(data_path, image_size=(128, 128), augment_data=False, augment_size=100, normalized=True):
    """
    Preprocess the dataset for Retina Blood Vessel Segmentation.

    Parameters:
    - data_path (str): Path to the dataset containing 'train' and 'test' folders.
    - image_size (tuple): The target size to which images will be resized (width, height).
    - augment_data (bool): If True, apply data augmentation to the training data.
    - augment_size (int): The number of augmented samples to generate.

    Returns:
    - A tuple (X_train, X_val, X_test, y_train, y_val, y_test) of preprocessed image and mask arrays.
    """
    def load_images_and_masks(image_path, mask_path):
        images, masks = [], []
        for img_name in os.listdir(image_path):
            img_path = os.path.join(image_path, img_name)
            mask_path_full = os.path.join(mask_path, img_name)
            if not os.path.exists(mask_path_full):
                continue  # Ensure corresponding mask exists

            # Load and resize the image and mask
            with Image.open(img_path) as img:
                img_resized = img.resize(image_size)

                # Normalize pixel values to [0, 1]
                if normalized:
                    img_array = np.array(img_resized) / 255.0  
                else:
                    img_array = np.array(img_resized)

                # Add channel dimension for grayscale images
                if len(img_array.shape) == 2:  
                    img_array = img_array[..., np.newaxis]
                images.append(img_array)

            with Image.open(mask_path_full) as mask:
                mask_resized = mask.resize(image_size)

                # Normalize mask values to [0, 1]
                mask_array = np.array(mask_resized) / 255.0  

                # Add channel dimension for grayscale masks
                if len(mask_array.shape) == 2:  
                    mask_array = mask_array[..., np.newaxis]
                masks.append(mask_array)

        return np.array(images), np.array(masks)

    # Load training data
    train_image_path = os.path.join(data_path, 'train', 'image')
    train_mask_path = os.path.join(data_path, 'train', 'mask')
    X_train, y_train = load_images_and_masks(train_image_path, train_mask_path)


    # Load testing data (without augmentation)
    test_image_path = os.path.join(data_path, 'test', 'image')
    test_mask_path = os.path.join(data_path, 'test', 'mask')
    X_test, y_test = load_images_and_masks(test_image_path, test_mask_path)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    # Apply data augmentation only to the training data
    if augment_data:
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True
        )

        augmented_images, augmented_masks = [], []
        count = 0
        while count < augment_size:
            for img, mask in zip(X_train, y_train):
                img = img.reshape((1, *img.shape))  
                mask = mask.reshape((1, *mask.shape))  

                # Use the same seed for image and mask to ensure augmentation consistency
                seed = np.random.randint(1, 10000)
                img_gen = datagen.flow(img, batch_size=1, seed=seed)
                mask_gen = datagen.flow(mask, batch_size=1, seed=seed)

                augmented_images.append(img_gen.__next__()[0])  
                augmented_masks.append(mask_gen.__next__()[0])

                count += 1
                if count >= augment_size:
                    break

        # Convert lists to arrays and ensure consistent shapes
        augmented_images = np.array(augmented_images)
        augmented_masks = np.array(augmented_masks)

        # Concatenate the original and augmented data
        X_train = np.concatenate((X_train, augmented_images), axis=0)
        y_train = np.concatenate((y_train, augmented_masks), axis=0)

    return X_train, X_val, X_test, y_train, y_val, y_test