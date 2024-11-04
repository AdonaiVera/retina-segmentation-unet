import os
from PIL import Image

def analyze_dataset(data_path):
    """
    Analyze the dataset for Retina Blood Vessel Segmentation.

    Parameters:
    - data_path (str): Path to the dataset containing 'train' and 'test' folders.

    Returns:
    - A dictionary with answers to the dataset questions.
    """
    analysis = {}

    # 1) Count the number of data samples in the training and test sets
    train_image_path = os.path.join(data_path, 'train', 'image')
    num_train_samples = len([name for name in os.listdir(train_image_path) if name.endswith('.png')])
    analysis['num_train_samples'] = num_train_samples

    test_image_path = os.path.join(data_path, 'test', 'image')
    num_test_samples = len([name for name in os.listdir(test_image_path) if name.endswith('.png')])
    analysis['num_test_samples'] = num_test_samples

    # 2) Problem description
    analysis['problem_description'] = "This dataset addresses the problem of medical image segmentation, specifically segmenting retinal blood vessels in fundus images."

    # 3) Dimension range of the images in the training set
    image_dims = []
    for img_name in os.listdir(train_image_path):
        img_path = os.path.join(train_image_path, img_name)
        with Image.open(img_path) as img:
            image_dims.append(img.size)
    analysis['dimension_range'] = (min(image_dims), max(image_dims))

    # 4) Check for missing data (missing images or masks)
    mask_path = os.path.join(data_path, 'train', 'mask')
    num_masks = len([name for name in os.listdir(mask_path) if name.endswith('.png')])
    analysis['missing_data'] = num_train_samples != num_masks

    # 5) Label description
    analysis['label_description'] = "Each image is annotated with a binary mask where blood vessel pixels are labeled as 1, and background pixels as 0."

    # 6) Data split
    analysis['data_split'] = "Pre-defined split: 80% training and 20% testing, already reflected in the folder structure."

    # 7) Data pre-processing methods
    analysis['data_preprocessing'] = [
        "Resizing images to a uniform size",
        "Normalization of pixel values (e.g., scaling to [0, 1])",
        "Data augmentation techniques such as rotation, flipping, and zoom for better generalization",
        "Ensuring the augmented images respect the original 80/20 train-test split"
    ]

    return analysis

if __name__ == "__main__":
    data_path = "data/Data" 
    results = analyze_dataset(data_path)
    for key, value in results.items():
        print(f"{key}: {value}")
