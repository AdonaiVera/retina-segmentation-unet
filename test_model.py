import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from models.unet_trainer import UNetTrainer
from methods.preprocessing import preprocess_data
from sklearn.metrics import jaccard_score

def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-7)


def calculate_metrics(y_true, y_pred):
    """
    Calculate Dice coefficient and IoU (Jaccard index) for predicted and ground truth masks.
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Ensure both arrays are binary
    y_true_flat = (y_true_flat > 0.5).astype(np.uint8)
    y_pred_flat = (y_pred_flat > 0.5).astype(np.uint8)

    dice = dice_coefficient(y_true_flat, y_pred_flat)
    iou = jaccard_score(y_true_flat, y_pred_flat, average='binary')
    
    return dice, iou

def test_model(weights_dir, X_test, y_test, num_examples=4, normalization=True):
    """
    Load trained model weights from a directory, predict on the test dataset, and show examples.

    Parameters:
    - weights_dir (str): Directory containing trained model weight files.
    - X_test (numpy.ndarray): Test images.
    - y_test (numpy.ndarray): True masks for the test images.
    - num_examples (int): Number of example images to display.
    """
    # Get all model weight filenames
    weight_files = [f for f in os.listdir(weights_dir) if f.endswith('.keras') or f.endswith('.h5')]
    weight_files.sort() 

    # Dictionary to store metrics for each model    
    metrics_summary = {}

     # Display example images
    plt.figure(figsize=(20, 15))
    for i in range(num_examples):
        index = np.random.randint(0, len(X_test))
        plt.subplot(num_examples, len(weight_files) + 2, i * (len(weight_files) + 2) + 1)
        plt.imshow(X_test[index].squeeze(), cmap='gray')
        plt.title('Input')
        plt.axis('off')

        plt.subplot(num_examples, len(weight_files) + 2, i * (len(weight_files) + 2) + 2)
        plt.imshow(y_test[index].squeeze(), cmap='gray')
        plt.title('Label')
        plt.axis('off')

        # Iterate through each model and predict
        for j, weight_file in enumerate(weight_files):
            trainer = UNetTrainer(input_shape=X_test.shape[1:])
            num_layers = int(weight_file.split('_')[1].replace('layers', ''))
            base_filters = int(weight_file.split('_')[2].replace('filters', ''))
            dropout_rate = float(weight_file.split('_')[3].replace('dropout', ''))
            learning_rate = str(weight_file.split('_')[4].replace('lr', '').replace('.keras', ''))
            
            model = trainer.build_unet(num_layers=num_layers, base_filters=base_filters, dropout_rate=dropout_rate)
            model.load_weights(os.path.join(weights_dir, weight_file))

            # Make predictions
            y_pred = model.predict(np.expand_dims(X_test[index], axis=0))
            y_pred_bin = (y_pred > 0.5).astype(np.uint8).squeeze()

            # Calculate metrics with normalization
            dice_norm, iou_norm = calculate_metrics(y_test[index], y_pred_bin)

            # Store metrics for summary
            if normalization:
                metrics_summary[weight_file] = {
                    "Dice with Normalization": dice_norm,
                    "IoU with normalization": iou_norm,
                }
            else:
                metrics_summary[weight_file] = {
                    "Dice without Normalization": dice_norm,
                    "IoU without normalization": iou_norm,
                }

            plt.subplot(num_examples, len(weight_files) + 2, i * (len(weight_files) + 2) + 3 + j)
            plt.imshow(y_pred_bin, cmap='gray')
            weight_labels = f"L{num_layers}_F{base_filters}_D{dropout_rate}_LR{learning_rate}"
            plt.title(weight_labels)
            plt.axis('off')

    # Save the combined figure
    if not os.path.exists('figures'):
        os.makedirs('figures')
    plt.savefig('figures/model_comparison.png')
    plt.show()
    print("Model comparison plot saved as 'figures/model_comparison.png'.")

    # Print metric summary for each model
    for model_name, metrics in metrics_summary.items():
        print(f"\nModel: {model_name}")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

if __name__ == "__main__":
    data_path = "data/Data"  
    _, _, X_test, _, _, y_test = preprocess_data(data_path, augment_data=True, normalized=False)
    print(f"Testing set: {X_test.shape}, {y_test.shape}")

    # Path to the weights directory
    weights_dir = 'models/weights'

    # Run the test model function
    test_model(weights_dir, X_test, y_test)
