// Product Upload Interface JavaScript
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('productImageForm');
    const imageInput = document.getElementById('productImageInput');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const imagePreview = document.getElementById('imagePreview');
    const previewImg = document.getElementById('previewImg');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const resultsSection = document.getElementById('resultsSection');
    const errorSection = document.getElementById('errorSection');
    
    // Result elements
    const detectedClass = document.getElementById('detectedClass');
    const confidenceBar = document.getElementById('confidenceBar');
    const confidenceText = document.getElementById('confidenceText');
    const topPredictions = document.getElementById('topPredictions');
    const predictionsList = document.getElementById('predictionsList');
    const naturalLanguageResponse = document.getElementById('naturalLanguageResponse');
    const productsTableBody = document.getElementById('productsTableBody');
    const noProductsMessage = document.getElementById('noProductsMessage');
    const errorMessage = document.getElementById('errorMessage');

    // File input change handler
    imageInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            // Validate file type
            if (!file.type.startsWith('image/')) {
                showError('Please select a valid image file.');
                return;
            }

            // Validate file size (16MB)
            if (file.size > 16 * 1024 * 1024) {
                showError('File size must be less than 16MB.');
                return;
            }

            // Show preview
            const reader = new FileReader();
            reader.onload = function(e) {
                previewImg.src = e.target.result;
                imagePreview.style.display = 'block';
                imagePreview.classList.add('fade-in');
                
                // Update button state
                analyzeBtn.disabled = false;
                analyzeBtn.classList.remove('btn-outline-warning');
                analyzeBtn.classList.add('btn-warning');
            };
            reader.readAsDataURL(file);
        } else {
            hidePreview();
        }
    });

    // Drag and drop functionality
    const dropZone = document.querySelector('.card-body');
    
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        dropZone.classList.add('border-warning', 'bg-light');
    }

    function unhighlight(e) {
        dropZone.classList.remove('border-warning', 'bg-light');
    }

    dropZone.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;

        if (files.length > 0) {
            imageInput.files = files;
            imageInput.dispatchEvent(new Event('change'));
        }
    }

    // Form submission
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        if (!imageInput.files[0]) {
            showError('Please select a product image.');
            return;
        }

        showLoading();
        hideResults();
        hideError();

        try {
            const formData = new FormData();
            formData.append('product_image', imageInput.files[0]);

            const response = await fetch('/image-product-search', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Request failed');
            }

            displayResults(data);
        } catch (error) {
            console.error('Error:', error);
            showError(error.message || 'An error occurred while analyzing your image.');
        } finally {
            hideLoading();
        }
    });

    function hidePreview() {
        imagePreview.style.display = 'none';
        imagePreview.classList.remove('fade-in');
        analyzeBtn.disabled = true;
        analyzeBtn.classList.add('btn-outline-warning');
        analyzeBtn.classList.remove('btn-warning');
    }

    function showLoading() {
        loadingIndicator.style.display = 'block';
        loadingIndicator.classList.add('fade-in');
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Analyzing...';
        
        // Scroll to loading indicator
        loadingIndicator.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    function hideLoading() {
        loadingIndicator.style.display = 'none';
        loadingIndicator.classList.remove('fade-in');
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<i class="fas fa-search me-2"></i>Analyze Image & Find Similar Products';
    }

    function showResults() {
        resultsSection.style.display = 'block';
        resultsSection.classList.add('slide-up');
        
        // Scroll to results
        setTimeout(() => {
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);
    }

    function hideResults() {
        resultsSection.style.display = 'none';
        resultsSection.classList.remove('slide-up');
    }

    function showError(message) {
        errorMessage.textContent = message;
        errorSection.style.display = 'block';
        errorSection.classList.add('fade-in');
        
        // Scroll to error
        errorSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    function hideError() {
        errorSection.style.display = 'none';
        errorSection.classList.remove('fade-in');
    }

    function displayResults(data) {
        // Display detected class
        detectedClass.textContent = data.detected_class || 'Unknown';
        
        // Display confidence
        const confidence = (data.confidence || 0) * 100;
        confidenceBar.style.width = confidence + '%';
        confidenceText.textContent = confidence.toFixed(1) + '% confidence';
        
        // Set confidence color
        if (confidence >= 80) {
            confidenceBar.className = 'progress-bar bg-success';
        } else if (confidence >= 60) {
            confidenceBar.className = 'progress-bar bg-warning';
        } else {
            confidenceBar.className = 'progress-bar bg-danger';
        }

        // Display top predictions if available
        if (data.top_predictions && data.top_predictions.length > 1) {
            predictionsList.innerHTML = '';
            data.top_predictions.slice(1).forEach((prediction, index) => {
                const badge = document.createElement('span');
                badge.className = 'badge bg-secondary me-2 mb-1';
                badge.textContent = `${prediction.class} (${(prediction.confidence * 100).toFixed(1)}%)`;
                predictionsList.appendChild(badge);
            });
            topPredictions.style.display = 'block';
        } else {
            topPredictions.style.display = 'none';
        }

        // Display natural language response with typing effect
        typeText(naturalLanguageResponse, data.response || 'No response available.');

        // Clear previous results
        productsTableBody.innerHTML = '';

        if (data.products && data.products.length > 0) {
            // Populate products table with animation
            data.products.forEach((product, index) => {
                setTimeout(() => {
                    const row = document.createElement('tr');
                    row.style.opacity = '0';
                    row.style.transform = 'translateY(20px)';
                    
                    // Determine similarity color
                    const similarity = product.similarity_score || 0;
                    let badgeClass = 'bg-secondary';
                    if (similarity > 0.7) badgeClass = 'bg-success';
                    else if (similarity > 0.5) badgeClass = 'bg-warning';
                    else if (similarity > 0.3) badgeClass = 'bg-info';
                    
                    row.innerHTML = `
                        <td><code class="text-warning">${product.stock_code || 'N/A'}</code></td>
                        <td>
                            <strong>${product.description || 'N/A'}</strong>
                        </td>
                        <td>
                            <span class="text-success fw-bold">$${(product.unit_price || 0).toFixed(2)}</span>
                        </td>
                        <td>
                            <span class="badge bg-light text-dark">${product.country || 'N/A'}</span>
                        </td>
                        <td>
                            <span class="badge ${badgeClass}">
                                ${(similarity * 100).toFixed(1)}%
                            </span>
                        </td>
                    `;
                    
                    productsTableBody.appendChild(row);
                    
                    // Animate row appearance
                    setTimeout(() => {
                        row.style.transition = 'all 0.3s ease';
                        row.style.opacity = '1';
                        row.style.transform = 'translateY(0)';
                    }, 50);
                }, index * 100);
            });
            
            noProductsMessage.style.display = 'none';
        } else {
            noProductsMessage.style.display = 'block';
        }

        showResults();
    }

    function typeText(element, text, speed = 30) {
        element.textContent = '';
        let i = 0;
        
        function type() {
            if (i < text.length) {
                element.textContent += text.charAt(i);
                i++;
                setTimeout(type, speed);
            }
        }
        
        type();
    }

    // Initialize button state
    analyzeBtn.disabled = true;
    analyzeBtn.classList.add('btn-outline-warning');
});
