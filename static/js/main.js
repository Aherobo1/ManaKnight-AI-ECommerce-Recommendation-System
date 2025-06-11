// Main JavaScript for E-Commerce Product Recommendation System

// Global utilities and common functionality
document.addEventListener('DOMContentLoaded', function() {
    
    // Initialize tooltips if Bootstrap is available
    if (typeof bootstrap !== 'undefined') {
        var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }

    // Add smooth scrolling to all anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Add loading states to buttons
    document.querySelectorAll('button[type="submit"]').forEach(button => {
        const form = button.closest('form');
        if (form) {
            form.addEventListener('submit', function() {
                if (!button.disabled) {
                    const originalText = button.innerHTML;
                    button.dataset.originalText = originalText;
                    button.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Loading...';
                    button.disabled = true;
                    
                    // Re-enable after 10 seconds as fallback
                    setTimeout(() => {
                        if (button.disabled) {
                            button.innerHTML = button.dataset.originalText || originalText;
                            button.disabled = false;
                        }
                    }, 10000);
                }
            });
        }
    });

    // Add fade-in animation to cards
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    // Observe all cards for animation
    document.querySelectorAll('.card').forEach(card => {
        observer.observe(card);
    });

    // Add click tracking for analytics (if needed)
    document.querySelectorAll('.btn').forEach(button => {
        button.addEventListener('click', function() {
            const action = this.textContent.trim();
            const page = window.location.pathname;
            
            // Log interaction (could be sent to analytics service)
            console.log('Button clicked:', {
                action: action,
                page: page,
                timestamp: new Date().toISOString()
            });
        });
    });

    // Add keyboard navigation support
    document.addEventListener('keydown', function(e) {
        // Alt + H for home
        if (e.altKey && e.key === 'h') {
            window.location.href = '/';
        }
        
        // Alt + T for text query
        if (e.altKey && e.key === 't') {
            window.location.href = '/text-query';
        }
        
        // Alt + I for image query
        if (e.altKey && e.key === 'i') {
            window.location.href = '/image-query';
        }
        
        // Alt + P for product upload
        if (e.altKey && e.key === 'p') {
            window.location.href = '/product-upload';
        }
    });

    // Add responsive table wrapper
    document.querySelectorAll('table').forEach(table => {
        if (!table.closest('.table-responsive')) {
            const wrapper = document.createElement('div');
            wrapper.className = 'table-responsive';
            table.parentNode.insertBefore(wrapper, table);
            wrapper.appendChild(table);
        }
    });

    // Add copy to clipboard functionality for code elements
    document.querySelectorAll('code').forEach(code => {
        code.style.cursor = 'pointer';
        code.title = 'Click to copy';
        
        code.addEventListener('click', function() {
            navigator.clipboard.writeText(this.textContent).then(() => {
                // Show temporary feedback
                const originalText = this.textContent;
                this.textContent = 'Copied!';
                this.style.backgroundColor = '#d4edda';
                
                setTimeout(() => {
                    this.textContent = originalText;
                    this.style.backgroundColor = '';
                }, 1000);
            }).catch(err => {
                console.error('Failed to copy text: ', err);
            });
        });
    });

    // Add auto-hide for alerts
    document.querySelectorAll('.alert').forEach(alert => {
        if (alert.classList.contains('alert-success')) {
            setTimeout(() => {
                alert.style.transition = 'opacity 0.5s ease';
                alert.style.opacity = '0';
                setTimeout(() => {
                    alert.remove();
                }, 500);
            }, 5000);
        }
    });

    // Add progress indicator for file uploads
    document.querySelectorAll('input[type="file"]').forEach(input => {
        input.addEventListener('change', function() {
            const file = this.files[0];
            if (file) {
                // Create progress indicator
                let progressDiv = this.parentNode.querySelector('.upload-progress');
                if (!progressDiv) {
                    progressDiv = document.createElement('div');
                    progressDiv.className = 'upload-progress mt-2';
                    progressDiv.innerHTML = `
                        <div class="progress">
                            <div class="progress-bar" role="progressbar" style="width: 0%"></div>
                        </div>
                        <small class="text-muted">File selected: ${file.name}</small>
                    `;
                    this.parentNode.appendChild(progressDiv);
                }
                
                // Animate progress bar
                const progressBar = progressDiv.querySelector('.progress-bar');
                let width = 0;
                const interval = setInterval(() => {
                    width += 10;
                    progressBar.style.width = width + '%';
                    if (width >= 100) {
                        clearInterval(interval);
                        progressBar.classList.add('bg-success');
                    }
                }, 50);
            }
        });
    });

    // Add network status indicator
    function updateNetworkStatus() {
        const statusIndicator = document.getElementById('network-status');
        if (statusIndicator) {
            if (navigator.onLine) {
                statusIndicator.className = 'badge bg-success';
                statusIndicator.textContent = 'Online';
            } else {
                statusIndicator.className = 'badge bg-danger';
                statusIndicator.textContent = 'Offline';
            }
        }
    }

    window.addEventListener('online', updateNetworkStatus);
    window.addEventListener('offline', updateNetworkStatus);
    updateNetworkStatus();

    // Add performance monitoring
    if ('performance' in window) {
        window.addEventListener('load', function() {
            setTimeout(() => {
                const perfData = performance.getEntriesByType('navigation')[0];
                console.log('Page load performance:', {
                    loadTime: perfData.loadEventEnd - perfData.loadEventStart,
                    domContentLoaded: perfData.domContentLoadedEventEnd - perfData.domContentLoadedEventStart,
                    totalTime: perfData.loadEventEnd - perfData.fetchStart
                });
            }, 0);
        });
    }

    // Add error boundary for JavaScript errors
    window.addEventListener('error', function(e) {
        console.error('JavaScript error:', {
            message: e.message,
            filename: e.filename,
            lineno: e.lineno,
            colno: e.colno,
            error: e.error
        });
        
        // Show user-friendly error message
        const errorDiv = document.createElement('div');
        errorDiv.className = 'alert alert-warning alert-dismissible fade show position-fixed';
        errorDiv.style.cssText = 'top: 20px; right: 20px; z-index: 9999; max-width: 300px;';
        errorDiv.innerHTML = `
            <strong>Oops!</strong> Something went wrong. Please refresh the page.
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        document.body.appendChild(errorDiv);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (errorDiv.parentNode) {
                errorDiv.remove();
            }
        }, 5000);
    });

    // Add service worker registration for PWA capabilities
    if ('serviceWorker' in navigator) {
        navigator.serviceWorker.register('/static/js/sw.js')
            .then(registration => {
                console.log('Service Worker registered:', registration);
            })
            .catch(error => {
                console.log('Service Worker registration failed:', error);
            });
    }
});

// Utility functions
window.AppUtils = {
    // Format currency
    formatCurrency: function(amount, currency = 'USD') {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: currency
        }).format(amount);
    },

    // Format date
    formatDate: function(date) {
        return new Intl.DateTimeFormat('en-US', {
            year: 'numeric',
            month: 'long',
            day: 'numeric'
        }).format(new Date(date));
    },

    // Debounce function
    debounce: function(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },

    // Show toast notification
    showToast: function(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        toast.style.cssText = 'top: 20px; right: 20px; z-index: 9999; max-width: 300px;';
        toast.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        document.body.appendChild(toast);
        
        setTimeout(() => {
            if (toast.parentNode) {
                toast.remove();
            }
        }, 5000);
    },

    // Validate file
    validateFile: function(file, maxSize = 16 * 1024 * 1024, allowedTypes = ['image/']) {
        if (!file) return { valid: false, error: 'No file selected' };
        
        if (file.size > maxSize) {
            return { valid: false, error: 'File size too large' };
        }
        
        const isValidType = allowedTypes.some(type => file.type.startsWith(type));
        if (!isValidType) {
            return { valid: false, error: 'Invalid file type' };
        }
        
        return { valid: true };
    }
};
