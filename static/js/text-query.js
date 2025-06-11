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
                    else if (similarity > 0.5) badgeClass = 'bg-primary';
                    else if (similarity > 0.3) badgeClass = 'bg-warning';
                    
                    row.innerHTML = `
                        <td><code class="text-primary">${product.stock_code || 'N/A'}</code></td>
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
