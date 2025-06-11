"""
Web Scraper Service Module

Handles web scraping for product images and data collection.
"""

import os
import requests
import time
import csv
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse
import hashlib
from pathlib import Path

try:
    from bs4 import BeautifulSoup
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False
    print("BeautifulSoup not available. Install with: pip install beautifulsoup4")

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("Selenium not available. Install with: pip install selenium")


class WebScraper:
    """
    Web scraper service for collecting product images and data.
    """
    
    def __init__(self, download_dir: str = "data/scraped_images", delay: float = 1.0):
        """
        Initialize web scraper.
        
        Args:
            download_dir (str): Directory to save scraped images
            delay (float): Delay between requests in seconds
        """
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Statistics
        self.stats = {
            'images_downloaded': 0,
            'images_failed': 0,
            'products_scraped': 0
        }
    
    def scrape_product_images(self, product_list: List[str], 
                            images_per_product: int = 10) -> Dict[str, List[str]]:
        """
        Scrape product images from search engines.
        
        Args:
            product_list (List[str]): List of product names to search
            images_per_product (int): Number of images to download per product
            
        Returns:
            Dict[str, List[str]]: Dictionary mapping product names to image paths
        """
        if not BEAUTIFULSOUP_AVAILABLE:
            print("BeautifulSoup required for web scraping")
            return {}
        
        scraped_data = {}
        
        for product in product_list:
            print(f"Scraping images for: {product}")
            
            # Create product directory
            product_dir = self.download_dir / self._sanitize_filename(product)
            product_dir.mkdir(exist_ok=True)
            
            # Search for images
            image_urls = self._search_product_images(product, images_per_product)
            
            # Download images
            downloaded_paths = []
            for i, url in enumerate(image_urls):
                if i >= images_per_product:
                    break
                
                image_path = self._download_image(url, product_dir, f"{product}_{i+1}")
                if image_path:
                    downloaded_paths.append(str(image_path))
                
                time.sleep(self.delay)
            
            scraped_data[product] = downloaded_paths
            self.stats['products_scraped'] += 1
            
            print(f"Downloaded {len(downloaded_paths)} images for {product}")
        
        return scraped_data
    
    def _search_product_images(self, product_name: str, max_images: int = 10) -> List[str]:
        """
        Search for product images using a simple image search.
        
        Args:
            product_name (str): Product name to search
            max_images (int): Maximum number of image URLs to return
            
        Returns:
            List[str]: List of image URLs
        """
        # This is a simplified implementation
        # In a real scenario, you'd use proper APIs or more sophisticated scraping
        
        image_urls = []
        
        try:
            # Example: Search on a sample e-commerce site (replace with actual sites)
            search_urls = [
                f"https://example-ecommerce.com/search?q={product_name.replace(' ', '+')}"
            ]
            
            for search_url in search_urls:
                try:
                    response = self.session.get(search_url, timeout=10)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, 'html.parser')
                        
                        # Find image tags (adjust selectors based on actual sites)
                        img_tags = soup.find_all('img', limit=max_images)
                        
                        for img in img_tags:
                            src = img.get('src') or img.get('data-src')
                            if src:
                                # Convert relative URLs to absolute
                                full_url = urljoin(search_url, src)
                                if self._is_valid_image_url(full_url):
                                    image_urls.append(full_url)
                
                except Exception as e:
                    print(f"Error searching {search_url}: {e}")
                    continue
                
                if len(image_urls) >= max_images:
                    break
        
        except Exception as e:
            print(f"Error in image search: {e}")
        
        # Fallback: Generate some sample image URLs for demonstration
        if not image_urls:
            image_urls = self._generate_sample_image_urls(product_name, max_images)
        
        return image_urls[:max_images]
    
    def _generate_sample_image_urls(self, product_name: str, count: int) -> List[str]:
        """Generate sample image URLs for demonstration purposes."""
        # This is for demonstration - replace with actual scraping logic
        sample_urls = []
        
        # Use placeholder image services for demonstration
        for i in range(count):
            # Generate different sized placeholder images
            width = 300 + (i * 50)
            height = 300 + (i * 50)
            url = f"https://via.placeholder.com/{width}x{height}/0066CC/FFFFFF?text={product_name.replace(' ', '+')}"
            sample_urls.append(url)
        
        return sample_urls
    
    def _is_valid_image_url(self, url: str) -> bool:
        """Check if URL points to a valid image."""
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return False
            
            # Check file extension
            path = parsed.path.lower()
            valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
            
            return any(path.endswith(ext) for ext in valid_extensions)
        
        except Exception:
            return False
    
    def _download_image(self, url: str, save_dir: Path, filename: str) -> Optional[Path]:
        """
        Download image from URL.
        
        Args:
            url (str): Image URL
            save_dir (Path): Directory to save image
            filename (str): Base filename (extension will be added)
            
        Returns:
            Optional[Path]: Path to downloaded image or None if failed
        """
        try:
            response = self.session.get(url, timeout=10, stream=True)
            response.raise_for_status()
            
            # Determine file extension from content type or URL
            content_type = response.headers.get('content-type', '')
            if 'jpeg' in content_type or 'jpg' in content_type:
                ext = '.jpg'
            elif 'png' in content_type:
                ext = '.png'
            elif 'gif' in content_type:
                ext = '.gif'
            elif 'webp' in content_type:
                ext = '.webp'
            else:
                # Try to get extension from URL
                parsed_url = urlparse(url)
                path_ext = os.path.splitext(parsed_url.path)[1]
                ext = path_ext if path_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'] else '.jpg'
            
            # Create unique filename
            safe_filename = self._sanitize_filename(filename)
            file_path = save_dir / f"{safe_filename}{ext}"
            
            # Avoid overwriting existing files
            counter = 1
            while file_path.exists():
                file_path = save_dir / f"{safe_filename}_{counter}{ext}"
                counter += 1
            
            # Save image
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            self.stats['images_downloaded'] += 1
            return file_path
        
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            self.stats['images_failed'] += 1
            return None
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for filesystem compatibility."""
        # Remove or replace invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        # Limit length
        return filename[:100]
    
    def scrape_product_data(self, urls: List[str]) -> List[Dict[str, Any]]:
        """
        Scrape product data from e-commerce URLs.
        
        Args:
            urls (List[str]): List of product page URLs
            
        Returns:
            List[Dict]: List of product data dictionaries
        """
        if not BEAUTIFULSOUP_AVAILABLE:
            print("BeautifulSoup required for web scraping")
            return []
        
        products = []
        
        for url in urls:
            try:
                response = self.session.get(url, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Extract product data (adjust selectors based on actual sites)
                    product_data = {
                        'url': url,
                        'title': self._extract_text(soup, ['h1', '.product-title', '.title']),
                        'price': self._extract_text(soup, ['.price', '.cost', '.amount']),
                        'description': self._extract_text(soup, ['.description', '.product-desc']),
                        'image_url': self._extract_image_url(soup),
                        'scraped_at': time.strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    products.append(product_data)
                
                time.sleep(self.delay)
            
            except Exception as e:
                print(f"Error scraping {url}: {e}")
                continue
        
        return products
    
    def _extract_text(self, soup: BeautifulSoup, selectors: List[str]) -> str:
        """Extract text using multiple selectors."""
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                return element.get_text(strip=True)
        return ""
    
    def _extract_image_url(self, soup: BeautifulSoup) -> str:
        """Extract main product image URL."""
        selectors = [
            '.product-image img',
            '.main-image img',
            '.hero-image img',
            'img[alt*="product"]'
        ]
        
        for selector in selectors:
            img = soup.select_one(selector)
            if img:
                return img.get('src') or img.get('data-src') or ""
        
        return ""
    
    def create_training_dataset(self, scraped_data: Dict[str, List[str]], 
                              output_csv: str = "CNN_Model_Train_Data.csv") -> str:
        """
        Create training dataset CSV from scraped images.
        
        Args:
            scraped_data (Dict): Dictionary mapping product names to image paths
            output_csv (str): Output CSV filename
            
        Returns:
            str: Path to created CSV file
        """
        csv_path = self.download_dir / output_csv
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['product_name', 'image_path', 'class_id'])
            
            class_id = 0
            for product_name, image_paths in scraped_data.items():
                for image_path in image_paths:
                    if os.path.exists(image_path):
                        writer.writerow([product_name, image_path, class_id])
                class_id += 1
        
        print(f"Training dataset created: {csv_path}")
        return str(csv_path)
    
    def get_scraping_stats(self) -> Dict[str, Any]:
        """Get scraping statistics."""
        return {
            'images_downloaded': self.stats['images_downloaded'],
            'images_failed': self.stats['images_failed'],
            'products_scraped': self.stats['products_scraped'],
            'success_rate': (
                self.stats['images_downloaded'] / 
                (self.stats['images_downloaded'] + self.stats['images_failed'])
                if (self.stats['images_downloaded'] + self.stats['images_failed']) > 0 else 0
            )
        }
    
    def cleanup_downloads(self, min_file_size: int = 1024):
        """
        Clean up downloaded files (remove small/corrupted images).
        
        Args:
            min_file_size (int): Minimum file size in bytes
        """
        removed_count = 0
        
        for file_path in self.download_dir.rglob('*'):
            if file_path.is_file():
                try:
                    if file_path.stat().st_size < min_file_size:
                        file_path.unlink()
                        removed_count += 1
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")
        
        print(f"Removed {removed_count} small/corrupted files")


if __name__ == "__main__":
    # Test web scraper
    scraper = WebScraper()
    
    # Example usage
    products = ["wireless headphones", "laptop computer", "smartphone"]
    scraped_data = scraper.scrape_product_images(products, images_per_product=5)
    
    print(f"Scraping stats: {scraper.get_scraping_stats()}")
    
    # Create training dataset
    if scraped_data:
        csv_path = scraper.create_training_dataset(scraped_data)
        print(f"Training dataset created at: {csv_path}")
