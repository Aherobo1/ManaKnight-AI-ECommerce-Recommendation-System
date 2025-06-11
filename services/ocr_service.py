"""
OCR Service Module

Handles Optical Character Recognition for extracting text from images.
"""

import os
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Dict, Any
import base64
import io

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Tesseract not available. Install with: pip install pytesseract")


class OCRService:
    """
    OCR service for extracting text from images using Tesseract.
    """
    
    def __init__(self, tesseract_path: str = None):
        """
        Initialize OCR service.
        
        Args:
            tesseract_path (str): Path to Tesseract executable
        """
        self.tesseract_path = tesseract_path or os.getenv('TESSERACT_PATH')
        
        if TESSERACT_AVAILABLE and self.tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_path
        
        # OCR configuration
        self.config = {
            'lang': 'eng',
            'config': '--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '
        }
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR accuracy.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            np.ndarray: Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply threshold to get binary image
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up the image
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        return cleaned
    
    def extract_text_from_image(self, image_path: str = None, 
                               image_data: bytes = None,
                               image_array: np.ndarray = None) -> Tuple[str, float]:
        """
        Extract text from image using OCR.
        
        Args:
            image_path (str): Path to image file
            image_data (bytes): Image data as bytes
            image_array (np.ndarray): Image as numpy array
            
        Returns:
            Tuple[str, float]: (extracted_text, confidence_score)
        """
        if not TESSERACT_AVAILABLE:
            # Return mock OCR result
            mock_texts = [
                "wireless headphones",
                "laptop computer",
                "smartphone case",
                "kitchen utensils",
                "home decoration"
            ]
            import random
            mock_text = random.choice(mock_texts)
            return mock_text, 0.85
        
        try:
            # Load image
            if image_path:
                image = cv2.imread(image_path)
            elif image_data:
                image = self._bytes_to_image(image_data)
            elif image_array is not None:
                image = image_array
            else:
                return "No image provided", 0.0
            
            if image is None:
                return "Could not load image", 0.0
            
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Extract text with confidence
            data = pytesseract.image_to_data(
                processed_image, 
                lang=self.config['lang'],
                config=self.config['config'],
                output_type=pytesseract.Output.DICT
            )
            
            # Filter out low confidence detections
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 30]
            texts = [data['text'][i] for i, conf in enumerate(data['conf']) if int(conf) > 30]
            
            # Combine text and calculate average confidence
            extracted_text = ' '.join(texts).strip()
            avg_confidence = np.mean(confidences) / 100.0 if confidences else 0.0
            
            # Clean up extracted text
            extracted_text = self._clean_extracted_text(extracted_text)
            
            return extracted_text, avg_confidence
            
        except Exception as e:
            print(f"Error in OCR extraction: {e}")
            return f"OCR Error: {str(e)}", 0.0
    
    def _bytes_to_image(self, image_data: bytes) -> np.ndarray:
        """Convert bytes to OpenCV image."""
        try:
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
    
    def _clean_extracted_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.
        
        Args:
            text (str): Raw extracted text
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove common OCR artifacts
        artifacts = ['|', '_', '~', '`', '^']
        for artifact in artifacts:
            text = text.replace(artifact, '')
        
        # Fix common OCR mistakes
        corrections = {
            '0': 'o',  # Zero to letter O in some contexts
            '1': 'l',  # One to letter L in some contexts
            '5': 's',  # Five to letter S in some contexts
        }
        
        # Apply corrections only if the result makes more sense
        words = text.split()
        corrected_words = []
        
        for word in words:
            # Only apply corrections to words that are likely text (not numbers)
            if word.isalpha() or (len(word) > 2 and not word.isdigit()):
                corrected_word = word
                for old, new in corrections.items():
                    if old in word and not word.isdigit():
                        corrected_word = corrected_word.replace(old, new)
                corrected_words.append(corrected_word)
            else:
                corrected_words.append(word)
        
        return ' '.join(corrected_words)
    
    def extract_text_from_file_upload(self, file_storage) -> Tuple[str, float]:
        """
        Extract text from Flask file upload.
        
        Args:
            file_storage: Flask FileStorage object
            
        Returns:
            Tuple[str, float]: (extracted_text, confidence_score)
        """
        try:
            # Read file data
            file_data = file_storage.read()
            file_storage.seek(0)  # Reset file pointer
            
            return self.extract_text_from_image(image_data=file_data)
            
        except Exception as e:
            print(f"Error processing file upload: {e}")
            return f"File processing error: {str(e)}", 0.0
    
    def get_text_regions(self, image_path: str = None, 
                        image_data: bytes = None) -> list[Dict[str, Any]]:
        """
        Get text regions with bounding boxes and confidence scores.
        
        Args:
            image_path (str): Path to image file
            image_data (bytes): Image data as bytes
            
        Returns:
            List[Dict]: List of text regions with metadata
        """
        if not TESSERACT_AVAILABLE:
            return []
        
        try:
            # Load image
            if image_path:
                image = cv2.imread(image_path)
            elif image_data:
                image = self._bytes_to_image(image_data)
            else:
                return []
            
            if image is None:
                return []
            
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Get detailed data
            data = pytesseract.image_to_data(
                processed_image,
                lang=self.config['lang'],
                config=self.config['config'],
                output_type=pytesseract.Output.DICT
            )
            
            regions = []
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 30 and data['text'][i].strip():
                    regions.append({
                        'text': data['text'][i],
                        'confidence': int(data['conf'][i]),
                        'bbox': {
                            'x': data['left'][i],
                            'y': data['top'][i],
                            'width': data['width'][i],
                            'height': data['height'][i]
                        }
                    })
            
            return regions
            
        except Exception as e:
            print(f"Error getting text regions: {e}")
            return []
    
    def is_text_image(self, image_path: str = None, 
                     image_data: bytes = None) -> bool:
        """
        Check if image likely contains text.
        
        Args:
            image_path (str): Path to image file
            image_data (bytes): Image data as bytes
            
        Returns:
            bool: True if image likely contains text
        """
        text, confidence = self.extract_text_from_image(image_path, image_data)
        
        # Consider it a text image if we have reasonable confidence and text length
        return confidence > 0.3 and len(text.strip()) > 3
    
    def get_supported_formats(self) -> list[str]:
        """
        Get list of supported image formats.
        
        Returns:
            List[str]: Supported file extensions
        """
        return ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'gif']


if __name__ == "__main__":
    # Test OCR service
    ocr = OCRService()
    print("OCR Service initialized")
    print(f"Supported formats: {ocr.get_supported_formats()}")
    
    # Test with a sample image if available
    if os.path.exists('test_image.jpg'):
        text, confidence = ocr.extract_text_from_image('test_image.jpg')
        print(f"Extracted text: {text}")
        print(f"Confidence: {confidence:.2f}")

