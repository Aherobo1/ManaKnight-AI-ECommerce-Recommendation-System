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
from services import (
    DatabaseService,
    RecommendationEngine,
    OCRService,
    CNNModel,
    WebScraper,
    VectorDatabase
)

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
    recommendation_engine = RecommendationEngine(db_service)
    ocr_service = OCRService()
    cnn_model = CNNModel()
    web_scraper = WebScraper()
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
        products, response = recommendation_engine.get_recommendations(query, top_k=5)

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

        # Extract text using OCR
        extracted_text, confidence = ocr_service.extract_text_from_file_upload(image_file)

        if not extracted_text.strip():
            return jsonify({
                "products": [],
                "response": "No text could be extracted from the image. Please ensure the image contains clear, readable text.",
                "extracted_text": extracted_text,
                "confidence": confidence
            })

        logger.info(f"Extracted text: {extracted_text} (confidence: {confidence:.2f})")

        # Get recommendations based on extracted text
        products, response = recommendation_engine.get_recommendations(extracted_text, top_k=5)

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
            detected_class, confidence, top_predictions = cnn_model.predict_image(image_path=filepath)

            logger.info(f"Detected class: {detected_class} (confidence: {confidence:.2f})")

            # Get similar products based on detected class
            products, response = recommendation_engine.get_recommendations(detected_class, top_k=5)

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
            "cnn": "OK" if cnn_model and cnn_model.model else "NOT_LOADED",
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
        # Get database stats
        analytics = db_service.get_query_analytics(days=30)

        # Get vector database stats
        vector_stats = vector_db.get_index_stats()

        # Get model info
        model_info = cnn_model.get_model_info()

        return jsonify({
            "query_analytics": analytics,
            "vector_database": vector_stats,
            "cnn_model": model_info,
            "timestamp": os.popen('date').read().strip()
        })

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({"error": "Could not retrieve statistics"}), 500

@app.route('/api/scrape-products', methods=['POST'])
def scrape_products():
    """Endpoint to trigger product scraping."""
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400

        data = request.json
        product_list = data.get('products', [])
        images_per_product = data.get('images_per_product', 10)

        if not product_list:
            return jsonify({"error": "Product list cannot be empty"}), 400

        logger.info(f"Starting scraping for {len(product_list)} products")

        # Start scraping (this could be made async for better UX)
        scraped_data = web_scraper.scrape_product_images(product_list, images_per_product)

        # Create training dataset
        csv_path = web_scraper.create_training_dataset(scraped_data)

        # Get scraping statistics
        stats = web_scraper.get_scraping_stats()

        return jsonify({
            "message": "Scraping completed successfully",
            "scraped_products": len(scraped_data),
            "training_dataset": csv_path,
            "statistics": stats
        })

    except Exception as e:
        logger.error(f"Error in product scraping: {e}")
        return jsonify({"error": "Scraping failed"}), 500

@app.route('/api/train-model', methods=['POST'])
def train_model():
    """Endpoint to trigger CNN model training."""
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400

        data = request.json
        train_data_path = data.get('train_data_path', 'data/scraped_images')
        epochs = data.get('epochs', 50)
        batch_size = data.get('batch_size', 32)

        if not os.path.exists(train_data_path):
            return jsonify({"error": f"Training data path does not exist: {train_data_path}"}), 400

        logger.info(f"Starting model training with {epochs} epochs")

        # Train model (this should be made async for production)
        training_results = cnn_model.train_model(
            train_data_path=train_data_path,
            epochs=epochs,
            batch_size=batch_size
        )

        if 'error' in training_results:
            return jsonify({"error": training_results['error']}), 500

        return jsonify({
            "message": "Model training completed successfully",
            "results": training_results
        })

    except Exception as e:
        logger.error(f"Error in model training: {e}")
        return jsonify({"error": "Model training failed"}), 500

@app.route('/api/update-vectors', methods=['POST'])
def update_vectors():
    """Endpoint to update product vectors."""
    try:
        logger.info("Updating product vectors")

        # Update recommendation engine vectors
        recommendation_engine.update_vectors()

        return jsonify({"message": "Product vectors updated successfully"})

    except Exception as e:
        logger.error(f"Error updating vectors: {e}")
        return jsonify({"error": "Vector update failed"}), 500

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
