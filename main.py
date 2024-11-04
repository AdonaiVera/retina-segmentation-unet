from methods.preprocessing import preprocess_data
from models.unet_trainer import UNetTrainer

if __name__ == "__main__":
    data_path = "data/Data"  
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(data_path, augment_data=True, augment_size=100)
    print(f"Training set: {X_train.shape}, {y_train.shape}")
    print(f"Validation set: {X_val.shape}, {y_val.shape}")
    print(f"Testing set: {X_test.shape}, {y_test.shape}")

    # List of different configurations to test
    configurations = [
        #{'num_layers': 2, 'base_filters': 32, 'dropout_rate': 0.3, 'learning_rate': 1e-3},
        #{'num_layers': 3, 'base_filters': 64, 'dropout_rate': 0.4, 'learning_rate': 1e-3},
        #{'num_layers': 4, 'base_filters': 64, 'dropout_rate': 0.5, 'learning_rate': 1e-4},
        {'num_layers': 5, 'base_filters': 64, 'dropout_rate': 0.3, 'learning_rate': 1e-4},
    ]

    # List to store results
    results_table = []

    for config in configurations:
        print(f"\nTraining U-Net with {config['num_layers']} layers, base filters {config['base_filters']}, "
              f"dropout rate {config['dropout_rate']}, learning rate {config['learning_rate']:.0e}.")

        trainer = UNetTrainer(input_shape=(128, 128, 3))
        model = trainer.build_unet(
            num_layers=config['num_layers'],
            base_filters=config['base_filters'],
            dropout_rate=config['dropout_rate']
        )
        history = trainer.compile_and_train(
            model, X_train, y_train, X_val, y_val,
            num_layers=config['num_layers'],
            base_filters=config['base_filters'],
            dropout_rate=config['dropout_rate'],
            learning_rate=config['learning_rate'],
            epochs=50
        )

        # Evaluate the model on the test set
        scores = trainer.evaluate_model(model, X_test, y_test)
        results_table.append({
            'Model': f"U-Net {config['num_layers']} layers",
            'Dice': scores['Dice'],
            'IoU': scores['IoU']
        })

        # Plot training history for each configuration
        trainer.plot_training_history(
            history,
            num_layers=config['num_layers'],
            base_filters=config['base_filters'],
            dropout_rate=config['dropout_rate'],
            learning_rate=config['learning_rate']
        )

    # Print the results table
    print("\nResults Summary:")
    for result in results_table:
        print(f"Model: {result['Model']}, Dice: {result['Dice']:.4f}, IoU: {result['IoU']:.4f}")
