# E-Commerce Product Recommendation System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive AI-powered e-commerce product recommendation system that combines natural language processing, OCR technology, and computer vision to provide intelligent product recommendations through multiple input methods.

## 🚀 Features

- **Natural Language Queries**: Process customer queries in natural language and provide relevant product recommendations
- **OCR-Based Query Processing**: Extract and process handwritten queries from uploaded images
- **Image-Based Product Detection**: Identify products from images using custom CNN models
- **Vector Database Integration**: Efficient similarity search using Pinecone vector database
- **Web Scraping Capabilities**: Automated product image collection for model training
- **RESTful API**: Well-documented API endpoints for all functionalities
- **Responsive Web Interface**: User-friendly frontend for all interaction modes

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#-features)
- [Technology Stack](#-technology-stack)
- [Installation](#-installation)
- [API Documentation](#-api-documentation)
- [Project Structure](#-project-structure)
- [Usage Examples](#-usage-examples)
- [Development Status](#-development-status)
- [Module Details](#module-details)
- [Contributing](#-contributing)
- [License](#-license)

## 🛠 Technology Stack

### Backend
- **Framework**: Flask (Python web framework)
- **Vector Database**: Pinecone (for similarity search and recommendations)
- **OCR**: Tesseract (text extraction from images)
- **Machine Learning**: TensorFlow/Keras (custom CNN model development)
- **Web Scraping**: BeautifulSoup, Selenium
- **Image Processing**: OpenCV, PIL

### Frontend
- **HTML5/CSS3**: Responsive web interfaces
- **JavaScript**: Interactive user experience
- **Bootstrap**: UI components and styling

### Database & Storage
- **Vector Database**: Pinecone
- **File Storage**: Local filesystem (configurable for cloud storage)

### Development Tools
- **Package Management**: pip, requirements.txt
- **Version Control**: Git
- **Documentation**: Markdown

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Step-by-Step Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ds_task_1ab
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv

   # On Windows
   venv\Scripts\activate

   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_ENVIRONMENT=your_pinecone_environment
   FLASK_ENV=development
   FLASK_DEBUG=True
   ```

5. **Initialize the database**
   ```bash
   python -c "from services.database import init_db; init_db()"
   ```

6. **Run the application**
   ```bash
   python app.py
   ```

The application will be available at `http://localhost:5000`

## 📚 API Documentation

### Base URL
```
http://localhost:5000
```

### Endpoints

#### 1. Product Recommendation Service
**POST** `/product-recommendation`

Process natural language queries and return product recommendations.

**Request:**
```json
{
  "query": "I need wireless headphones for gaming"
}
```

**Response:**
```json
{
  "products": [
    {
      "stock_code": "001",
      "description": "High-Quality Gaming Headphones",
      "unit_price": 89.99,
      "country": "USA",
      "similarity_score": 0.95
    }
  ],
  "response": "I found excellent gaming headphones that match your requirements...",
  "query_processed": "wireless headphones gaming"
}
```

#### 2. OCR-Based Query Processing
**POST** `/ocr-query`

Extract text from handwritten images and process as product queries.

**Request:**
- Form data with `image_data` file upload

**Response:**
```json
{
  "products": [...],
  "response": "Based on your handwritten query...",
  "extracted_text": "wireless mouse for office work",
  "confidence": 0.87
}
```

#### 3. Image-Based Product Detection
**POST** `/image-product-search`

Identify products from uploaded images using CNN model.

**Request:**
- Form data with `product_image` file upload

**Response:**
```json
{
  "products": [...],
  "response": "I identified this as a smartphone...",
  "detected_class": "smartphone",
  "confidence": 0.92
}
```

#### 4. Sample Response
**GET** `/sample_response`

Returns a sample HTML response showing the expected output format.

## 📁 Project Structure

```
ds_task_1ab/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── .env                  # Environment variables (create this)
├── data/                 # Data storage
│   └── dataset.zip       # E-commerce dataset
├── services/             # Backend services
│   ├── __init__.py
│   ├── database.py       # Database operations
│   ├── recommendation.py # Recommendation engine
│   ├── ocr_service.py    # OCR functionality
│   ├── cnn_model.py      # CNN model for image detection
│   └── scraper.py        # Web scraping utilities
├── models/               # Trained models
│   └── cnn_product_classifier.h5
├── templates/            # HTML templates
│   ├── sample_response.html
│   ├── text_query.html
│   ├── image_query.html
│   └── product_upload.html
├── static/               # Static files (CSS, JS, images)
│   ├── css/
│   ├── js/
│   └── images/
├── notebooks/            # Jupyter notebooks for development
│   ├── data_cleaning.ipynb
│   ├── model_training.ipynb
│   └── vector_database_setup.ipynb
└── tests/                # Unit tests
    ├── test_api.py
    ├── test_services.py
    └── test_models.py
```

## 🎯 Usage Examples

### 1. Text Query Interface
```python
import requests

response = requests.post('http://localhost:5000/product-recommendation',
                        data={'query': 'affordable laptop for students'})
print(response.json())
```

### 2. Image Upload for OCR
```python
import requests

with open('handwritten_query.jpg', 'rb') as f:
    response = requests.post('http://localhost:5000/ocr-query',
                           files={'image_data': f})
print(response.json())
```

### 3. Product Image Detection
```python
import requests

with open('product_image.jpg', 'rb') as f:
    response = requests.post('http://localhost:5000/image-product-search',
                           files={'product_image': f})
print(response.json())
```

## 🚧 Development Status

### ✅ Completed
- [x] Basic Flask application structure
- [x] API endpoint definitions
- [x] Sample response template
- [x] Project documentation framework

### 🔄 In Progress
- [ ] Data cleaning and preprocessing
- [ ] Vector database integration
- [ ] OCR implementation
- [ ] CNN model development
- [ ] Web scraping functionality

### 📋 Planned
- [ ] Frontend interfaces
- [ ] Unit tests
- [ ] Performance optimization
- [ ] Deployment configuration

# Project Overview

This project is divided into four main modules, each focusing on a distinct aspect of the system's development. The modules are designed to work together seamlessly, culminating in a comprehensive solution for product recommendation, OCR-based query processing, and image-based product detection.

## Module 1: Data Preparation and Backend Setup

### Task 1: E-commerce Dataset Cleaning

- *Objective*: Ensure the dataset is clean and ready for analysis and vectorization.
- *Key Actions*: Remove duplicates, handle missing values, and standardize formats.

### Task 2: Vector Database Creation

- *Objective*: Set up a vector database using Pinecone to store product vectors.
- *Key Actions*: Define the database schema and integrate with Pinecone.

### Task 3: Similarity Metrics Selection

- *Objective*: Choose and justify the similarity metrics used to compare product vectors.
- *Key Actions*: Evaluate different metrics (e.g., cosine similarity, dot product) and select the best fit based on the dataset characteristics.

### Endpoint 1: Product Recommendation Service

- *Functionality*: Handle natural language queries to recommend products, including safeguards against bad queries and sensitive data exposure.
- *Input*: Customer's natural language query.
- *Output*: Product matches array and a natural language response within specified constraints.

## Module 2: OCR and Web Scraping

### Task 4: OCR Functionality Implementation

- *Objective*: Develop the capability to extract text from images using OCR technology.
- *Key Actions*: Integrate and configure an OCR tool (e.g., Tesseract).

### Task 5: Web Scraping for Product Images

- *Objective*: Scrape product images from e-commerce websites for training data ``CNN_Model_Train_Data.csv``.
- *Key Actions*: Automate scraping, download images, and store them systematically and make sure you have enough data to train the CNN model.

### Endpoint 2: OCR-Based Query Processing

- *Functionality*: Extract and process handwritten queries using the same logic as Endpoint 1.
- *Input*: Image file with handwritten text.
- *Output*: Same output format as Endpoint 1, adapted for image inputs also return the extracted test from OCR.

## Module 3: CNN Model Development

### Task 6: CNN Model Training

- *Objective*: Develop a CNN model from scratch using only the ``products`` mentioned on ``CNN_Model_Train_Data.csv`` to identify products from images.
- *Key Actions*: Train the model using scraped images and clean data without using pre-trained models.

### Endpoint 3: Image-Based Product Detection

- *Functionality*: Use the CNN model to identify products from images and match them using the vector database.
- *Input*: Product image.
- *Output*: Product description and matching products in a format consistent with other endpoints. Also return the name of the `class` that you got from CNN model for the particular input image.

## Module 4: Frontend Development and Integration

### Frontend Page 1: Text Query Interface

- *Features*: Form to submit text queries, display natural language responses, and a product details table.

### Frontend Page 2: Image Query Interface

- *Features*: Allows users to upload images of handwritten queries and displays results similar to Page 1.

### Frontend Page 3: Product Image Upload Interface

- *Features*: Users can upload product images, and view the identified product description and related products in natural language and tabular format.

## Instructions for Presentation

### 1. Incremental Report Writing

Each module completion should be accompanied by a concise, to-the-point report that documents the process, decisions, and outcomes. These reports will be incremental, building upon each other as the bootcamp progresses.

#### Report Format Suggestion:

- *Title Page*: Include the module number and title, the names of the team members, and the submission date.
- *Introduction*: Briefly describe the objectives of the module and its importance to the overall project.
- *High-Level Flow*:
  - *Description*: Outline the main tasks and functionalities developed in the module.
  - *Diagrams*: Include flowcharts or diagrams that visually represent the architecture and data flow.
  - *Key Decisions*: Summarize crucial decisions made during the module, such as choice of technology, design patterns, and configurations.
- *Challenges and Solutions*:
  - Briefly discuss any challenges faced during the module and how they were addressed.
- *Conclusion*: Sum up the outcomes of the module and its readiness for integration with other modules.
- *References*: Cite any tools, libraries, or external resources that were used.

### 2. Video Documentation

Participants are required to create two sets of videos for each module, detailing both the functionality and the technical implementation. This will not only aid in a better understanding of the project but also serve as a reference for future projects.

#### Video Requirements:

- *Functional Demonstration Video*:
  - *Content*: Demonstrate the functionality of each endpoint and page developed in the module.
  - *Focus*: Show how the system responds to various inputs and scenarios. Explain the user interaction with the system.
  - *Duration*: Keep the video concise, preferably under 5 minutes.
- *Code Explanation Video*:
  - *Content*: Provide a high-level overview of the codebase for the module.
  - *Focus*: Explain the structure of the code, major classes, and functions. Highlight any significant patterns or algorithms used.
  - *Duration*: Limit the explanation to under 10 minutes.

### Submission Guidelines:

- *Timing*: Submit the videos along with the incremental report at the end of each module.
- *Format*: Ensure videos are in a common format (e.g., MP4) and quality is sufficient for clear viewing.
- *Hosting*: Upload videos to a platform accessible to all participants and reviewers (e.g., Google Drive, YouTube in unlisted mode). Or you can use loom, fluvid, vmaker etc alternatively.

## 🧪 Testing

### Running Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_api.py

# Run with coverage
python -m pytest --cov=services tests/
```

### Test Structure
- `tests/test_api.py` - API endpoint tests
- `tests/test_services.py` - Service layer tests
- `tests/test_models.py` - Model functionality tests

## 🚀 Deployment

### Local Development
```bash
export FLASK_ENV=development
python app.py
```

### Production Deployment
```bash
# Using Gunicorn
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:app

# Using Docker (create Dockerfile first)
docker build -t ecommerce-recommendation .
docker run -p 8000:8000 ecommerce-recommendation
```

### Environment Variables for Production
```env
FLASK_ENV=production
PINECONE_API_KEY=your_production_api_key
PINECONE_ENVIRONMENT=your_production_environment
SECRET_KEY=your_secret_key
DATABASE_URL=your_database_url
```

## 📊 Performance Metrics

### Expected Performance
- **Query Response Time**: < 500ms for text queries
- **OCR Processing**: < 2s for standard images
- **CNN Inference**: < 1s for product classification
- **Vector Search**: < 100ms for similarity matching

### Monitoring
- API response times
- Database query performance
- Model inference latency
- Error rates and exceptions

## 🔧 Configuration

### Pinecone Setup
1. Create account at [Pinecone](https://www.pinecone.io/)
2. Create an index with appropriate dimensions
3. Add API key to environment variables

### OCR Configuration
```python
# Tesseract configuration
TESSERACT_CONFIG = {
    'lang': 'eng',
    'config': '--psm 6'
}
```

### CNN Model Configuration
```python
# Model parameters
MODEL_CONFIG = {
    'input_shape': (224, 224, 3),
    'num_classes': 50,
    'batch_size': 32,
    'epochs': 100
}
```

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### Development Workflow
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`python -m pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Code Style
- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings for all functions and classes
- Keep functions small and focused

### Commit Messages
- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit first line to 72 characters
- Reference issues and pull requests when applicable

## 📝 Changelog

### Version 1.0.0 (In Development)
- Initial project setup
- Basic API structure
- Documentation framework
- Sample response templates

## 🐛 Known Issues

- OCR accuracy may vary with handwriting quality
- CNN model requires sufficient training data
- Vector database initialization may take time on first run

## 📞 Support

For support and questions:
- Create an issue in the repository
- Contact the development team
- Check the documentation for common solutions

## 🙏 Acknowledgments

- Pinecone for vector database services
- Tesseract OCR community
- Flask development team
- Open source contributors

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Team

- **Project Lead**: [Your Name]
- **Backend Developer**: [Team Member]
- **ML Engineer**: [Team Member]
- **Frontend Developer**: [Team Member]

---

**Note**: This project is part of a data science bootcamp and is designed for educational purposes. The system demonstrates various AI/ML techniques including NLP, computer vision, and recommendation systems.

## Instructions for Coding

### General Guidelines

- *Class-Based Implementation*: It is recommended to use class-based implementation for all backend services to ensure organized, reusable, and maintainable code.
- *Best Practices*:
  - *ACID Properties*: Ensure that database transactions are Atomic, Consistent, Isolated, and Durable to maintain data integrity and reliability.
  - *Modularity*: Build the codebase with clear modularity in mind. Separate different functionalities into distinct modules to enhance readability and maintainability.
- *Packaging*: Organize your code into packages that reflect the services they provide. This approach not only helps in maintaining the code but also simplifies the deployment and scaling process.
- Directories: Whenever you will test on notebook make sure you keep all the notebooks in ``notebook`` directory and use proper naming for the notebooks.

### Tech Stack

- *Web Framework*: Use Flask for developing the backend. Flask provides flexibility and ease of use for setting up API services.
- *Vector Database*: Integrate Pinecone to manage and query vector data efficiently. Pinecone supports scalable vector searches which are crucial for the recommendation systems in this project.
