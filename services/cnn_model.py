"""
CNN Model Service Module

Handles CNN model for image-based product classification.
"""

import os
import numpy as np
import cv2
from PIL import Image
from typing import Tuple, List, Dict, Any, Optional
import pickle
import json

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Install with: pip install tensorflow")


class CNNModel:
    """
    CNN model service for product image classification.
    """
    
    def __init__(self, model_path: str = "models/cnn_product_classifier.h5"):
        """
        Initialize CNN model service.

        Args:
            model_path (str): Path to the trained model file
        """
        self.model_path = model_path
        self.model = None
        self.class_names = [
            'electronics', 'clothing', 'home_decor', 'kitchen', 'toys',
            'books', 'sports', 'beauty', 'automotive', 'jewelry'
        ]
        self.input_shape = (224, 224, 3)
        self.num_classes = len(self.class_names)

        # Load model if it exists, otherwise create a simple one
        if os.path.exists(model_path):
            self.load_model()
        else:
            print(f"Model not found at {model_path}. Creating simple model...")
            self.create_simple_model()
    
    def create_model(self, num_classes: int = None, input_shape: Tuple = None) -> keras.Model:
        """
        Create CNN model architecture.
        
        Args:
            num_classes (int): Number of product classes
            input_shape (Tuple): Input image shape
            
        Returns:
            keras.Model: CNN model
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for CNN model")
        
        num_classes = num_classes or self.num_classes
        input_shape = input_shape or self.input_shape
        
        model = keras.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Global Average Pooling
            layers.GlobalAveragePooling2D(),
            
            # Dense Layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output Layer
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_5_accuracy']
        )
        
        return model

    def create_simple_model(self):
        """Create a simple CNN model for demonstration."""
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available - using mock predictions")
            return

        try:
            self.model = self.create_model(self.num_classes, self.input_shape)
            print("✅ Simple CNN model created")
        except Exception as e:
            print(f"Error creating model: {e}")
            self.model = None
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for model prediction.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            np.ndarray: Preprocessed image
        """
        # Resize to model input size
        image = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))
        
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values
        image = image.astype(np.float32) / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def predict_image(self, image_path: str = None, 
                     image_data: bytes = None,
                     image_array: np.ndarray = None) -> Tuple[str, float, List[Dict[str, Any]]]:
        """
        Predict product class from image.
        
        Args:
            image_path (str): Path to image file
            image_data (bytes): Image data as bytes
            image_array (np.ndarray): Image as numpy array
            
        Returns:
            Tuple[str, float, List[Dict]]: (predicted_class, confidence, top_predictions)
        """
        if self.model is None:
            # Return mock prediction if no model
            import random
            mock_class = random.choice(self.class_names)
            mock_confidence = random.uniform(0.7, 0.95)
            mock_predictions = [
                {'class': mock_class, 'confidence': mock_confidence, 'rank': 1},
                {'class': random.choice(self.class_names), 'confidence': random.uniform(0.1, 0.3), 'rank': 2},
                {'class': random.choice(self.class_names), 'confidence': random.uniform(0.05, 0.15), 'rank': 3}
            ]
            return mock_class, mock_confidence, mock_predictions
        
        try:
            # Load image
            if image_path:
                image = cv2.imread(image_path)
            elif image_data:
                image = self._bytes_to_image(image_data)
            elif image_array is not None:
                image = image_array
            else:
                return "No image provided", 0.0, []
            
            if image is None:
                return "Could not load image", 0.0, []
            
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            
            # Get top predictions
            top_indices = np.argsort(predictions[0])[::-1][:5]
            top_predictions = []
            
            for i, idx in enumerate(top_indices):
                class_name = self.class_names[idx] if idx < len(self.class_names) else f"Class_{idx}"
                confidence = float(predictions[0][idx])
                
                top_predictions.append({
                    'class': class_name,
                    'confidence': confidence,
                    'rank': i + 1
                })
            
            # Return best prediction
            best_class = top_predictions[0]['class'] if top_predictions else "Unknown"
            best_confidence = top_predictions[0]['confidence'] if top_predictions else 0.0
            
            return best_class, best_confidence, top_predictions
            
        except Exception as e:
            print(f"Error in image prediction: {e}")
            return f"Prediction Error: {str(e)}", 0.0, []
    
    def _bytes_to_image(self, image_data: bytes) -> np.ndarray:
        """Convert bytes to OpenCV image."""
        try:
            import io
            import base64
            
            # Try to decode as base64 first
            if isinstance(image_data, str):
                image_data = base64.b64decode(image_data)
            
            # Convert to PIL Image then to OpenCV
            pil_image = Image.open(io.BytesIO(image_data))
            opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            return opencv_image
            
        except Exception as e:
            print(f"Error converting bytes to image: {e}")
            return None
    
    def train_model(self, train_data_path: str, validation_split: float = 0.2,
                   epochs: int = 50, batch_size: int = 32) -> Dict[str, Any]:
        """
        Train the CNN model.
        
        Args:
            train_data_path (str): Path to training data directory
            validation_split (float): Fraction of data for validation
            epochs (int): Number of training epochs
            batch_size (int): Training batch size
            
        Returns:
            Dict: Training history and metrics
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for model training")
        
        try:
            # Create data generators
            train_datagen = keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                zoom_range=0.2,
                validation_split=validation_split
            )
            
            train_generator = train_datagen.flow_from_directory(
                train_data_path,
                target_size=(self.input_shape[1], self.input_shape[0]),
                batch_size=batch_size,
                class_mode='categorical',
                subset='training'
            )
            
            validation_generator = train_datagen.flow_from_directory(
                train_data_path,
                target_size=(self.input_shape[1], self.input_shape[0]),
                batch_size=batch_size,
                class_mode='categorical',
                subset='validation'
            )
            
            # Update class information
            self.num_classes = train_generator.num_classes
            self.class_names = list(train_generator.class_indices.keys())
            
            # Create model
            self.model = self.create_model(self.num_classes, self.input_shape)
            
            # Callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=10,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=5,
                    min_lr=0.0001
                )
            ]
            
            # Train model
            history = self.model.fit(
                train_generator,
                epochs=epochs,
                validation_data=validation_generator,
                callbacks=callbacks,
                verbose=1
            )
            
            # Save model and class names
            self.save_model()
            
            return {
                'history': history.history,
                'final_accuracy': max(history.history['val_accuracy']),
                'num_classes': self.num_classes,
                'class_names': self.class_names
            }
            
        except Exception as e:
            print(f"Error training model: {e}")
            return {'error': str(e)}
    
    def save_model(self):
        """Save the trained model and metadata."""
        if self.model is None:
            print("No model to save")
            return
        
        try:
            # Create models directory
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Save model
            self.model.save(self.model_path)
            
            # Save class names and metadata
            metadata = {
                'class_names': self.class_names,
                'num_classes': self.num_classes,
                'input_shape': self.input_shape
            }
            
            metadata_path = self.model_path.replace('.h5', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
            
            print(f"Model saved to {self.model_path}")
            
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self):
        """Load the trained model and metadata."""
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available - cannot load model")
            return
        
        try:
            # Load model
            self.model = keras.models.load_model(self.model_path)
            
            # Load metadata
            metadata_path = self.model_path.replace('.h5', '_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                self.class_names = metadata.get('class_names', [])
                self.num_classes = metadata.get('num_classes', 50)
                self.input_shape = tuple(metadata.get('input_shape', [224, 224, 3]))
            
            print(f"Model loaded from {self.model_path}")
            print(f"Classes: {len(self.class_names)}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def evaluate_model(self, test_data_path: str) -> Dict[str, Any]:
        """
        Evaluate model performance on test data.
        
        Args:
            test_data_path (str): Path to test data directory
            
        Returns:
            Dict: Evaluation metrics
        """
        if self.model is None:
            return {'error': 'Model not loaded'}
        
        try:
            test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
            
            test_generator = test_datagen.flow_from_directory(
                test_data_path,
                target_size=(self.input_shape[1], self.input_shape[0]),
                batch_size=32,
                class_mode='categorical',
                shuffle=False
            )
            
            # Evaluate
            results = self.model.evaluate(test_generator, verbose=0)
            
            return {
                'test_loss': results[0],
                'test_accuracy': results[1],
                'test_top5_accuracy': results[2] if len(results) > 2 else None
            }
            
        except Exception as e:
            print(f"Error evaluating model: {e}")
            return {'error': str(e)}
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dict: Model information
        """
        if self.model is None:
            return {'status': 'No model loaded'}
        
        return {
            'status': 'Model loaded',
            'num_classes': self.num_classes,
            'input_shape': self.input_shape,
            'class_names': self.class_names[:10],  # First 10 classes
            'total_params': self.model.count_params(),
            'model_path': self.model_path
        }


if __name__ == "__main__":
    # Test CNN model service
    cnn = CNNModel()
    info = cnn.get_model_info()
    print(f"CNN Model Info: {info}")
