// Text Query Interface JavaScript
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('textQueryForm');
    const queryInput = document.getElementById('queryInput');
    const searchBtn = document.getElementById('searchBtn');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const resultsSection = document.getElementById('resultsSection');
    const errorSection = document.getElementById('errorSection');
    const naturalLanguageResponse = document.getElementById('naturalLanguageResponse');
    const productsTableBody = document.getElementById('productsTableBody');
    const productsGrid = document.getElementById('productsGrid');
    const noProductsMessage = document.getElementById('noProductsMessage');
    const errorMessage = document.getElementById('errorMessage');

    // Example query buttons
    document.querySelectorAll('.example-query').forEach(button => {
        button.addEventListener('click', function() {
            queryInput.value = this.dataset.query;
            queryInput.focus();
            // Auto-submit after a short delay
            setTimeout(() => {
                form.dispatchEvent(new Event('submit'));
            }, 300);
        });
    });

    // Auto-focus on input
    queryInput.focus();

    // Form submission
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const query = queryInput.value.trim();
        if (!query) {
            showError('Please enter a search query.');
            queryInput.focus();
            return;
        }

        showLoading();
        hideResults();
        hideError();

        try {
            const formData = new FormData();
            formData.append('query', query);

            const response = await fetch('/product-recommendation', {
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
            showError(error.message || 'An error occurred while processing your request.');
        } finally {
            hideLoading();
        }
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + Enter to submit
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            form.dispatchEvent(new Event('submit'));
        }
        // Escape to clear
        if (e.key === 'Escape') {
            queryInput.value = '';
            hideResults();
            hideError();
            queryInput.focus();
        }
    });

    function showLoading() {
        loadingIndicator.style.display = 'block';
        loadingIndicator.classList.add('fade-in');
        searchBtn.disabled = true;
        searchBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Searching...';
        
        // Scroll to loading indicator
        loadingIndicator.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    function hideLoading() {
        loadingIndicator.style.display = 'none';
        loadingIndicator.classList.remove('fade-in');
        searchBtn.disabled = false;
        searchBtn.innerHTML = '<i class="fas fa-search me-2"></i>Search';
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
        // Display natural language response with typing effect
        typeText(naturalLanguageResponse, data.response || 'No response available.');

        // Clear previous results
        productsGrid.innerHTML = '';
        productsTableBody.innerHTML = '';

        if (data.products && data.products.length > 0) {
            // Populate products grid with beautiful cards
            data.products.forEach((product, index) => {
                setTimeout(() => {
                    // Determine similarity color
                    const similarity = product.similarity_score || 0;
                    let badgeClass = 'bg-secondary';
                    let badgeText = 'Low Match';
                    if (similarity > 0.9) { badgeClass = 'bg-success'; badgeText = 'Excellent Match'; }
                    else if (similarity > 0.7) { badgeClass = 'bg-primary'; badgeText = 'Good Match'; }
                    else if (similarity > 0.5) { badgeClass = 'bg-warning'; badgeText = 'Fair Match'; }

                    // Create product card
                    const cardCol = document.createElement('div');
                    cardCol.className = 'col-md-6 col-lg-4 mb-4';
                    cardCol.style.opacity = '0';
                    cardCol.style.transform = 'translateY(30px)';

                    const imagePath = product.image_path || '/static/images/products/default.png';

                    cardCol.innerHTML = `
                        <div class="card product-card h-100">
                            <div class="position-relative">
                                <img src="${imagePath}"
                                     class="card-img-top"
                                     alt="${product.description || 'Product'}"
                                     style="height: 200px; object-fit: cover;"
                                     onerror="this.src='/static/images/products/default.png'">
                                <div class="position-absolute top-0 end-0 m-2">
                                    <span class="badge ${badgeClass} px-3 py-2" style="border-radius: 20px;">
                                        ${(similarity * 100).toFixed(1)}% ${badgeText}
                                    </span>
                                </div>
                            </div>
                            <div class="card-body d-flex flex-column">
                                <div class="mb-2">
                                    <small class="text-muted fw-bold">${product.stock_code || 'N/A'}</small>
                                </div>
                                <h6 class="card-title text-dark mb-3" style="line-height: 1.4;">
                                    ${product.description || 'N/A'}
                                </h6>
                                <div class="mt-auto">
                                    <div class="d-flex justify-content-between align-items-center">
                                        <div>
                                            <span class="h5 text-success fw-bold mb-0">
                                                $${(product.unit_price || 0).toFixed(2)}
                                            </span>
                                        </div>
                                        <div>
                                            <small class="text-muted">
                                                <i class="fas fa-globe me-1"></i>
                                                ${product.country || 'N/A'}
                                            </small>
                                        </div>
                                    </div>
                                    <div class="mt-3">
                                        <button class="btn btn-primary btn-sm w-100" style="border-radius: 20px;">
                                            <i class="fas fa-shopping-cart me-2"></i>
                                            View Details
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    `;

                    productsGrid.appendChild(cardCol);

                    // Animate card appearance
                    setTimeout(() => {
                        cardCol.style.transition = 'all 0.5s ease';
                        cardCol.style.opacity = '1';
                        cardCol.style.transform = 'translateY(0)';
                    }, 100);
                }, index * 150);
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

    // Add some visual feedback for form interaction
    queryInput.addEventListener('input', function() {
        if (this.value.length > 0) {
            searchBtn.classList.add('btn-primary');
            searchBtn.classList.remove('btn-outline-primary');
        } else {
            searchBtn.classList.remove('btn-primary');
            searchBtn.classList.add('btn-outline-primary');
        }
    });

    // Initialize button state
    if (queryInput.value.length === 0) {
        searchBtn.classList.add('btn-outline-primary');
        searchBtn.classList.remove('btn-primary');
    }
});
