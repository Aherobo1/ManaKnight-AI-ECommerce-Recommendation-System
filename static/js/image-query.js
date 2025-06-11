// Image Query Interface JavaScript
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('imageQueryForm');
    const imageInput = document.getElementById('imageInput');
    const uploadBtn = document.getElementById('uploadBtn');
    const imagePreview = document.getElementById('imagePreview');
    const previewImg = document.getElementById('previewImg');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const resultsSection = document.getElementById('resultsSection');
    const errorSection = document.getElementById('errorSection');
    
    // Result elements
    const extractedText = document.getElementById('extractedText');
    const confidenceBar = document.getElementById('confidenceBar');
    const confidenceText = document.getElementById('confidenceText');
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
                uploadBtn.disabled = false;
                uploadBtn.classList.remove('btn-outline-success');
                uploadBtn.classList.add('btn-success');
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
        dropZone.classList.add('border-success', 'bg-light');
    }

    function unhighlight(e) {
        dropZone.classList.remove('border-success', 'bg-light');
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
            showError('Please select an image file.');
            return;
        }

        showLoading();
        hideResults();
        hideError();

        try {
            const formData = new FormData();
            formData.append('image_data', imageInput.files[0]);

            const response = await fetch('/ocr-query', {
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
            showError(error.message || 'An error occurred while processing your image.');
        } finally {
            hideLoading();
        }
    });

    function hidePreview() {
        imagePreview.style.display = 'none';
        imagePreview.classList.remove('fade-in');
        uploadBtn.disabled = true;
        uploadBtn.classList.add('btn-outline-success');
        uploadBtn.classList.remove('btn-success');
    }

    function showLoading() {
        loadingIndicator.style.display = 'block';
        loadingIndicator.classList.add('fade-in');
        uploadBtn.disabled = true;
        uploadBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
        
        // Scroll to loading indicator
        loadingIndicator.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    function hideLoading() {
        loadingIndicator.style.display = 'none';
        loadingIndicator.classList.remove('fade-in');
        uploadBtn.disabled = false;
        uploadBtn.innerHTML = '<i class="fas fa-magic me-2"></i>Extract Text & Search';
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
        // Display extracted text
        extractedText.textContent = data.extracted_text || 'No text extracted';
        
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
                    else if (similarity > 0.5) badgeClass = 'bg-info';
                    else if (similarity > 0.3) badgeClass = 'bg-warning';
                    
                    row.innerHTML = `
                        <td><code class="text-info">${product.stock_code || 'N/A'}</code></td>
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
    uploadBtn.disabled = true;
    uploadBtn.classList.add('btn-outline-success');
});
