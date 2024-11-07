import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import matplotlib.pyplot as plt
import time
import os

class UNetTrainer:
    def __init__(self, input_shape=(128, 128, 3), num_classes=1):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_unet(self, num_layers=4, base_filters=64, dropout_rate=0.5):
        """
        Build a U-Net model with a customizable number of layers and filters.

        Parameters:
        - num_layers (int): Number of downsampling/upsampling layers.
        - base_filters (int): Number of filters for the first convolutional block.
        - dropout_rate (float): Dropout rate for regularization.

        Returns:
        - model (tf.keras.Model): A compiled U-Net model.
        """
        inputs = tf.keras.Input(shape=self.input_shape)
        skip_connections = []
        x = inputs

        # Encoder
        for i in range(num_layers):
            filters = base_filters * (2 ** i)
            x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
            x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
            skip_connections.append(x)
            x = layers.MaxPooling2D((2, 2))(x)
            if dropout_rate > 0:
                x = layers.Dropout(dropout_rate)(x)

        # Bottleneck
        x = layers.Conv2D(filters * 2, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(filters * 2, (3, 3), activation='relu', padding='same')(x)

        # Decoder
        for i in reversed(range(num_layers)):
            filters = base_filters * (2 ** i)
            x = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)
            x = layers.concatenate([x, skip_connections.pop()])
            x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
            x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
            if dropout_rate > 0:
                x = layers.Dropout(dropout_rate)(x)

        outputs = layers.Conv2D(self.num_classes, (1, 1), activation='sigmoid')(x)

        model = Model(inputs=[inputs], outputs=[outputs])
        return model

    @tf.function
    def dice_coefficient(self, y_true, y_pred, smooth=1e-6):
        """
        Calculate the Dice coefficient.

        Parameters:
        - y_true: Ground truth tensor.
        - y_pred: Prediction tensor.
        - smooth: Smoothing factor to avoid division by zero.

        Returns:
        - Dice coefficient score.
        """
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

    def compile_and_train(self, model, X_train, y_train, X_val, y_val, num_layers, base_filters, dropout_rate,
                          loss_function='binary_crossentropy', optimizer='adam', learning_rate=1e-3, batch_size=8,
                          epochs=50, model_dir='models/weights'):
        """
        Compile and train the U-Net model, saving the best model with the configuration in the filename and tracking training time.

        Parameters:
        - model (tf.keras.Model): The U-Net model to train.
        - X_train, y_train, X_val, y_val: Training and validation data.
        - num_layers (int): Number of downsampling/upsampling layers.
        - base_filters (int): Number of filters for the first convolutional block.
        - dropout_rate (float): Dropout rate for regularization.
        - loss_function (str): Loss function for training.
        - optimizer (str): Optimizer to use ('adam', 'sgd', 'rmsprop').
        - learning_rate (float): Learning rate for the optimizer.
        - batch_size (int): Batch size for training.
        - epochs (int): Number of epochs for training.
        - model_dir (str): Directory to save the best model.

        Returns:
        - history (tf.keras.callbacks.History): The history of the training process.
        """
        # Create model save path with configuration details
        model_filename = f"{model_dir}/unet_layers{num_layers}_filters{base_filters}_dropout{dropout_rate}_lr{learning_rate:.0e}.keras"

        # Choose the optimizer
        if optimizer == 'adam':
            opt = Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer == 'rmsprop':
            opt = RMSprop(learning_rate=learning_rate)
        else:
            raise ValueError("Unsupported optimizer. Choose 'adam', 'sgd', or 'rmsprop'.")

        # Compile the model
        model.compile(optimizer=opt, loss=loss_function, metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2), self.dice_coefficient])

        # Callbacks for avoiding overfitting and saving the best model
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            model_filename, monitor='val_loss', save_best_only=True, verbose=1
        )

        # Print the start time
        start_time = time.time()
        print("Training started at:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

        # Train the model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr, model_checkpoint]
        )

        # Print the end time and total duration
        end_time = time.time()
        print("Training ended at:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
        print("Total training time: {:.2f} seconds".format(end_time - start_time))

        return history

    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate the trained U-Net model on the test set and print IoU and Dice scores.

        Parameters:
        - model (tf.keras.Model): The trained U-Net model.
        - X_test, y_test: Test data.

        Returns:
        - Dictionary containing IoU and Dice scores.
        """
        results = model.evaluate(X_test, y_test, verbose=1)
        iou = results[2]  
        dice = results[3] 

        print(f"Test IoU: {iou:.4f}, Test Dice: {dice:.4f}")
        return {'IoU': iou, 'Dice': dice}

    def plot_training_history(self, history, num_layers, base_filters, dropout_rate, learning_rate, save_dir='figures'):
        """
        Plot the training and validation loss and accuracy and save them to a file.

        Parameters:
        - history (tf.keras.callbacks.History): The history of the training process.
        - num_layers (int): Number of downsampling/upsampling layers.
        - base_filters (int): Number of filters for the first convolutional block.
        - dropout_rate (float): Dropout rate for regularization.
        - learning_rate (float): Learning rate for the optimizer.
        - save_dir (str): Directory to save the plot images.
        """
        # Create the directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Create the filename with configuration details
        filename = f"{save_dir}/training_history_layers{num_layers}_filters{base_filters}_dropout{dropout_rate}_lr{learning_rate:.0e}.png"

        plt.figure(figsize=(12, 4))

        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        # Save the plot to a file
        plt.savefig(filename)
        plt.close()
        print(f"Training history plot saved as {filename}")