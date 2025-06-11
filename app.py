import os
import logging
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from flask_cors import CORS
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import traceback

# Load environment variables
load_dotenv()

# Import services
from services.database import DatabaseService
from services.recommendation import RecommendationEngine
from services.ocr_service import OCRService
from services.scraper import ProductImageScraper
from services.vector_db import VectorDatabase
from services.enhanced_cnn_model import EnhancedCNNModel

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_FILE_SIZE', 16777216))  # 16MB
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Enable CORS
CORS(app)

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Initialize services
try:
    db_service = DatabaseService()
    recommendation_engine = RecommendationEngine()
    ocr_service = OCRService()
    cnn_model = EnhancedCNNModel()
    web_scraper = ProductImageScraper()
    vector_db = VectorDatabase()

    logger.info("All services initialized successfully")
except Exception as e:
    logger.error(f"Error initializing services: {e}")
    # Continue with limited functionality

# Allowed file extensions
ALLOWED_EXTENSIONS = set(os.getenv('ALLOWED_EXTENSIONS', 'jpg,jpeg,png,gif,bmp,tiff').split(','))

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Home page with navigation to different interfaces."""
    return render_template('index.html')

@app.route('/text-query')
def text_query_page():
    """Text query interface page."""
    return render_template('text_query.html')

@app.route('/image-query')
def image_query_page():
    """Image query interface page."""
    return render_template('image_query.html')

@app.route('/product-upload')
def product_upload_page():
    """Product image upload interface page."""
    return render_template('product_upload.html')

@app.route('/product-recommendation', methods=['POST'])
def product_recommendation():
    """
    Endpoint for product recommendations based on natural language queries.
    Input: Form data containing 'query' (string).
    Output: JSON with 'products' (array of objects) and 'response' (string).
    """
    try:
        # Get query from form data or JSON
        if request.is_json:
            query = request.json.get('query', '')
        else:
            query = request.form.get('query', '')

        if not query.strip():
            return jsonify({
                "error": "Query cannot be empty",
                "products": [],
                "response": "Please provide a search query."
            }), 400

        logger.info(f"Processing text query: {query}")

        # Get recommendations
        result = recommendation_engine.get_recommendations(query)
        products = result.get('products', [])
        response = result.get('response', 'Found matching products for your query.')

        # Format products for response
        formatted_products = []
        for product in products:
            formatted_products.append({
                "stock_code": product.get('stock_code', ''),
                "description": product.get('description', ''),
                "unit_price": float(product.get('unit_price', 0)),
                "country": product.get('country', ''),
                "similarity_score": float(product.get('similarity_score', 0))
            })

        return jsonify({
            "products": formatted_products,
            "response": response,
            "query_processed": query.strip()
        })

    except Exception as e:
        logger.error(f"Error in product recommendation: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": "Internal server error",
            "products": [],
            "response": "Sorry, there was an error processing your request. Please try again."
        }), 500

@app.route('/ocr-query', methods=['POST'])
def ocr_query():
    """
    Endpoint to process handwritten queries extracted from uploaded images.
    Input: Form data containing 'image_data' (file, base64-encoded image or direct file upload).
    Output: JSON with 'products' (array of objects) and 'response' (string).
    """
    try:
        if 'image_data' not in request.files:
            return jsonify({
                "error": "No image file provided",
                "products": [],
                "response": "Please upload an image containing handwritten text.",
                "extracted_text": "",
                "confidence": 0.0
            }), 400

        image_file = request.files['image_data']

        if image_file.filename == '':
            return jsonify({
                "error": "No file selected",
                "products": [],
                "response": "Please select an image file.",
                "extracted_text": "",
                "confidence": 0.0
            }), 400

        if not allowed_file(image_file.filename):
            return jsonify({
                "error": "Invalid file type",
                "products": [],
                "response": "Please upload a valid image file (jpg, png, gif, etc.).",
                "extracted_text": "",
                "confidence": 0.0
            }), 400

        logger.info(f"Processing OCR query from file: {image_file.filename}")

        # Save file temporarily for OCR processing
        filename = secure_filename(image_file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(temp_path)

        try:
            # Extract text using OCR
            result = ocr_service.extract_text_from_image(temp_path)
            extracted_text = result.get('extracted_text', '')
            confidence = result.get('confidence', 0.0)
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

        if not extracted_text.strip():
            return jsonify({
                "products": [],
                "response": "No text could be extracted from the image. Please ensure the image contains clear, readable text.",
                "extracted_text": extracted_text,
                "confidence": confidence
            })

        logger.info(f"Extracted text: {extracted_text} (confidence: {confidence:.2f})")

        # Get recommendations based on extracted text
        result = recommendation_engine.get_recommendations(extracted_text)
        products = result.get('products', [])
        response = result.get('response', 'Found matching products for your query.')

        # Format products for response
        formatted_products = []
        for product in products:
            formatted_products.append({
                "stock_code": product.get('stock_code', ''),
                "description": product.get('description', ''),
                "unit_price": float(product.get('unit_price', 0)),
                "country": product.get('country', ''),
                "similarity_score": float(product.get('similarity_score', 0))
            })

        # Enhance response with OCR information
        enhanced_response = f"Based on your handwritten query '{extracted_text}': {response}"

        return jsonify({
            "products": formatted_products,
            "response": enhanced_response,
            "extracted_text": extracted_text,
            "confidence": confidence
        })

    except Exception as e:
        logger.error(f"Error in OCR query processing: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": "Internal server error",
            "products": [],
            "response": "Sorry, there was an error processing your image. Please try again.",
            "extracted_text": "",
            "confidence": 0.0
        }), 500

@app.route('/image-product-search', methods=['POST'])
def image_product_search():
    """
    Endpoint to identify and suggest products from uploaded product images.
    Input: Form data containing 'product_image' (file, base64-encoded image or direct file upload).
    Output: JSON with 'products' (array of objects) and 'response' (string).
    """
    try:
        if 'product_image' not in request.files:
            return jsonify({
                "error": "No image file provided",
                "products": [],
                "response": "Please upload a product image.",
                "detected_class": "",
                "confidence": 0.0
            }), 400

        image_file = request.files['product_image']

        if image_file.filename == '':
            return jsonify({
                "error": "No file selected",
                "products": [],
                "response": "Please select an image file.",
                "detected_class": "",
                "confidence": 0.0
            }), 400

        if not allowed_file(image_file.filename):
            return jsonify({
                "error": "Invalid file type",
                "products": [],
                "response": "Please upload a valid image file (jpg, png, gif, etc.).",
                "detected_class": "",
                "confidence": 0.0
            }), 400

        logger.info(f"Processing product image: {image_file.filename}")

        # Save uploaded file temporarily
        filename = secure_filename(image_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(filepath)

        try:
            # Classify image using CNN model
            prediction_result = cnn_model.predict_product_category(filepath)
            detected_class = prediction_result['predicted_class']
            confidence = prediction_result['confidence']
            top_predictions = prediction_result.get('top_predictions', [])

            logger.info(f"Detected class: {detected_class} (confidence: {confidence:.2f})")

            # Get similar products based on detected class
            result = recommendation_engine.get_recommendations(detected_class)
            products = result.get('products', [])
            response = result.get('response', 'Found matching products for your query.')

            # Format products for response
            formatted_products = []
            for product in products:
                formatted_products.append({
                    "stock_code": product.get('stock_code', ''),
                    "description": product.get('description', ''),
                    "unit_price": float(product.get('unit_price', 0)),
                    "country": product.get('country', ''),
                    "similarity_score": float(product.get('similarity_score', 0))
                })

            # Enhance response with classification information
            enhanced_response = f"I identified this as a {detected_class} with {confidence:.1%} confidence. {response}"

            return jsonify({
                "products": formatted_products,
                "response": enhanced_response,
                "detected_class": detected_class,
                "confidence": confidence,
                "top_predictions": top_predictions[:3]  # Top 3 predictions
            })

        finally:
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)

    except Exception as e:
        logger.error(f"Error in image product search: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": "Internal server error",
            "products": [],
            "response": "Sorry, there was an error processing your image. Please try again.",
            "detected_class": "",
            "confidence": 0.0
        }), 500

@app.route('/sample_response', methods=['GET'])
def sample_response():
    """
    Endpoint to return a sample JSON response for the API.
    Output: HTML template showing expected response format.
    """
    return render_template('sample_response.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring."""
    try:
        # Check database connection
        db_status = "OK" if db_service else "ERROR"

        # Check services status
        services_status = {
            "database": db_status,
            "recommendation": "OK" if recommendation_engine else "ERROR",
            "ocr": "OK" if ocr_service else "ERROR",
            "cnn": "OK" if cnn_model else "ERROR",
            "vector_db": "OK" if vector_db else "ERROR"
        }

        return jsonify({
            "status": "healthy",
            "services": services_status,
            "timestamp": os.popen('date').read().strip()
        })

    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics."""
    try:
        # Get basic stats
        products = db_service.get_all_products()

        return jsonify({
            "total_products": len(products),
            "categories": 10,
            "api_version": "1.0",
            "uptime": "Running",
            "timestamp": os.popen('date').read().strip()
        })

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({"error": "Could not retrieve statistics"}), 500



@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return jsonify({
        "error": "File too large",
        "message": f"File size exceeds the maximum limit of {app.config['MAX_CONTENT_LENGTH']} bytes"
    }), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {e}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Initialize database with sample data if needed
    try:
        # Check if we have products in database
        products = db_service.get_products(limit=1)
        if not products:
            logger.info("No products found. Consider loading sample data.")

            # Try to load from dataset.zip if it exists
            if os.path.exists('data/dataset.zip'):
                logger.info("Found dataset.zip - consider extracting and loading data")

    except Exception as e:
        logger.error(f"Error checking database: {e}")

    # Run the application
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'

    logger.info(f"Starting Flask application on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
