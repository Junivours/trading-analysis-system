"""
Performance monitoring utilities
"""
import time
import functools
from typing import Dict, Any
import logging

logger = logging.getLogger('PerformanceMonitor')

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {}
    
    def time_function(self, func_name: str):
        """Decorator to time function execution"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    
                    # Store metrics
                    if func_name not in self.metrics:
                        self.metrics[func_name] = []
                    
                    self.metrics[func_name].append(execution_time)
                    
                    # Log slow functions
                    if execution_time > 1.0:  # Log if takes more than 1 second
                        logger.warning(f"{func_name} took {execution_time:.2f}s to execute")
                    
                    return result
                except Exception as e:
                    logger.error(f"Error in {func_name}: {str(e)}")
                    raise
            return wrapper
        return decorator
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report"""
        report = {}
        
        for func_name, times in self.metrics.items():
            if times:
                report[func_name] = {
                    'calls': len(times),
                    'avg_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'total_time': sum(times)
                }
        
        return report

# Global performance monitor instance
performance_monitor = PerformanceMonitor()
