"""
📊 Performance Monitoring Service - Mana Knight Digital

Real-time performance monitoring and metrics collection.
Features: Response time tracking, error rate monitoring, resource usage.
"""

import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    🎯 Professional performance monitoring service.
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize performance monitor.
        
        Args:
            max_history (int): Maximum number of metrics to keep in memory
        """
        self.max_history = max_history
        
        # Metrics storage
        self.request_times = deque(maxlen=max_history)
        self.error_counts = defaultdict(int)
        self.endpoint_stats = defaultdict(lambda: {
            'count': 0,
            'total_time': 0,
            'avg_time': 0,
            'min_time': float('inf'),
            'max_time': 0,
            'errors': 0
        })
        
        # System metrics
        self.system_metrics = deque(maxlen=100)
        
        # Start background monitoring
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_system, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Performance monitor initialized")
    
    def record_request(self, endpoint: str, response_time: float, status_code: int):
        """
        Record a request for monitoring.
        
        Args:
            endpoint (str): API endpoint
            response_time (float): Response time in seconds
            status_code (int): HTTP status code
        """
        timestamp = datetime.now()
        
        # Record request time
        self.request_times.append({
            'timestamp': timestamp,
            'endpoint': endpoint,
            'response_time': response_time,
            'status_code': status_code
        })
        
        # Update endpoint statistics
        stats = self.endpoint_stats[endpoint]
        stats['count'] += 1
        stats['total_time'] += response_time
        stats['avg_time'] = stats['total_time'] / stats['count']
        stats['min_time'] = min(stats['min_time'], response_time)
        stats['max_time'] = max(stats['max_time'], response_time)
        
        # Record errors
        if status_code >= 400:
            stats['errors'] += 1
            self.error_counts[status_code] += 1
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics.
        
        Returns:
            Dict[str, Any]: Performance metrics
        """
        now = datetime.now()
        
        # Calculate recent metrics (last 5 minutes)
        recent_requests = [
            req for req in self.request_times
            if now - req['timestamp'] <= timedelta(minutes=5)
        ]
        
        # Calculate overall statistics
        total_requests = len(self.request_times)
        recent_count = len(recent_requests)
        
        avg_response_time = 0
        error_rate = 0
        
        if total_requests > 0:
            avg_response_time = sum(req['response_time'] for req in self.request_times) / total_requests
            error_count = sum(1 for req in self.request_times if req['status_code'] >= 400)
            error_rate = (error_count / total_requests) * 100
        
        # Get system metrics
        system_info = self._get_current_system_metrics()
        
        return {
            'timestamp': now.isoformat(),
            'requests': {
                'total': total_requests,
                'recent_5min': recent_count,
                'avg_response_time': round(avg_response_time, 3),
                'error_rate_percent': round(error_rate, 2)
            },
            'endpoints': dict(self.endpoint_stats),
            'errors': dict(self.error_counts),
            'system': system_info,
            'uptime': self._get_uptime()
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get system health status.
        
        Returns:
            Dict[str, Any]: Health status
        """
        metrics = self.get_performance_metrics()
        
        # Determine health status
        status = "healthy"
        issues = []
        
        # Check response time
        if metrics['requests']['avg_response_time'] > 2.0:
            status = "warning"
            issues.append("High average response time")
        
        # Check error rate
        if metrics['requests']['error_rate_percent'] > 5.0:
            status = "warning" if status == "healthy" else "critical"
            issues.append("High error rate")
        
        # Check system resources
        if metrics['system']['cpu_percent'] > 80:
            status = "warning" if status == "healthy" else "critical"
            issues.append("High CPU usage")
        
        if metrics['system']['memory_percent'] > 85:
            status = "warning" if status == "healthy" else "critical"
            issues.append("High memory usage")
        
        return {
            'status': status,
            'issues': issues,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
    
    def _monitor_system(self):
        """Background system monitoring."""
        while self.monitoring:
            try:
                metrics = self._get_current_system_metrics()
                self.system_metrics.append({
                    'timestamp': datetime.now(),
                    **metrics
                })
                time.sleep(30)  # Update every 30 seconds
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _get_current_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        try:
            return {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent,
                'network_io': {
                    'bytes_sent': psutil.net_io_counters().bytes_sent,
                    'bytes_recv': psutil.net_io_counters().bytes_recv
                }
            }
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {
                'cpu_percent': 0,
                'memory_percent': 0,
                'disk_percent': 0,
                'network_io': {'bytes_sent': 0, 'bytes_recv': 0}
            }
    
    def _get_uptime(self) -> str:
        """Get system uptime."""
        try:
            uptime_seconds = time.time() - psutil.boot_time()
            uptime_hours = int(uptime_seconds // 3600)
            uptime_minutes = int((uptime_seconds % 3600) // 60)
            return f"{uptime_hours}h {uptime_minutes}m"
        except:
            return "Unknown"
    
    def get_endpoint_performance(self, endpoint: str) -> Dict[str, Any]:
        """
        Get performance metrics for specific endpoint.
        
        Args:
            endpoint (str): Endpoint name
            
        Returns:
            Dict[str, Any]: Endpoint performance metrics
        """
        if endpoint not in self.endpoint_stats:
            return {'error': 'Endpoint not found'}
        
        stats = self.endpoint_stats[endpoint]
        
        # Get recent requests for this endpoint
        now = datetime.now()
        recent_requests = [
            req for req in self.request_times
            if req['endpoint'] == endpoint and 
               now - req['timestamp'] <= timedelta(minutes=5)
        ]
        
        return {
            'endpoint': endpoint,
            'total_requests': stats['count'],
            'recent_requests_5min': len(recent_requests),
            'avg_response_time': round(stats['avg_time'], 3),
            'min_response_time': round(stats['min_time'], 3) if stats['min_time'] != float('inf') else 0,
            'max_response_time': round(stats['max_time'], 3),
            'error_count': stats['errors'],
            'error_rate_percent': round((stats['errors'] / stats['count']) * 100, 2) if stats['count'] > 0 else 0
        }
    
    def reset_metrics(self):
        """Reset all metrics."""
        self.request_times.clear()
        self.error_counts.clear()
        self.endpoint_stats.clear()
        self.system_metrics.clear()
        logger.info("Performance metrics reset")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.monitoring = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        logger.info("Performance monitoring stopped")


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


def monitor_request(endpoint: str):
    """
    Decorator to monitor request performance.
    
    Args:
        endpoint (str): Endpoint name
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            status_code = 200
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status_code = 500
                raise
            finally:
                end_time = time.time()
                response_time = end_time - start_time
                performance_monitor.record_request(endpoint, response_time, status_code)
        
        return wrapper
    return decorator


if __name__ == "__main__":
    # Test performance monitor
    monitor = PerformanceMonitor()
    
    # Simulate some requests
    monitor.record_request('/test', 0.123, 200)
    monitor.record_request('/test', 0.156, 200)
    monitor.record_request('/test', 0.089, 404)
    
    # Get metrics
    metrics = monitor.get_performance_metrics()
    print(f"Performance metrics: {metrics}")
    
    health = monitor.get_health_status()
    print(f"Health status: {health}")
    
    monitor.stop_monitoring()
