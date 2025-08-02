#!/usr/bin/env python3
"""
RTX 3050 GPU Optimization Module
Optimizes memory usage and parallel processing for 6GB VRAM
"""

import os
import gc
import time
from typing import List, Dict, Optional, Callable
import threading
from queue import Queue
import multiprocessing

class RTX3050Optimizer:
    """
    GPU optimization specifically tuned for RTX 3050 6GB
    Manages memory efficiently and enables GPU acceleration where beneficial
    """
    
    def __init__(self):
        """Initialize RTX 3050 optimizations"""
        self.gpu_available = self._check_gpu()
        self.memory_limit = 4096  # Conservative 4GB limit for 6GB card
        self.max_workers = min(3, multiprocessing.cpu_count())  # Conservative threading
        
        # Memory management
        self._setup_memory_optimization()
        
        print(f"RTX 3050 Optimizer initialized")
        print(f" GPU Available: {self.gpu_available}")
        print(f" Memory Limit: {self.memory_limit}MB")
        print(f" Max Workers: {self.max_workers}")
    
    def _check_gpu(self) -> bool:
        """Check if GPU is available and suitable"""
        try:
            # Try to detect NVIDIA GPU
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0 and 'RTX 3050' in result.stdout:
                print(" RTX 3050 detected!")
                return True
            else:
                print("ℹ️ RTX 3050 not detected, using CPU optimization")
                return False
        except:
            print("ℹ️ GPU detection failed, using CPU optimization")
            return False
    
    def _setup_memory_optimization(self):
        """Setup memory optimization for 6GB VRAM"""
        if self.gpu_available:
            try:
                # Set environment variables for GPU memory optimization
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
                os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
                os.environ['TF_GPU_MEMORY_LIMIT'] = str(self.memory_limit)
                
                # PyTorch memory optimization
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.set_per_process_memory_fraction(0.7)  # Use 70% of VRAM
                        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
                        print(" PyTorch GPU optimizations enabled")
                except ImportError:
                    pass
                    
            except Exception as e:
                print(f"⚠️ GPU optimization setup failed: {e}")
    
    def optimize_memory(self):
        """Optimize memory usage and clean up GPU memory"""
        try:
            # Python garbage collection
            gc.collect()
            
            # GPU memory cleanup
            if self.gpu_available:
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                except ImportError:
                    pass
            
            # Force memory cleanup
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            if memory_info.rss > self.memory_limit * 1024 * 1024:  # Convert MB to bytes
                print(f"⚠️ High memory usage detected: {memory_info.rss / 1024 / 1024:.1f}MB")
                gc.collect()
                
        except Exception as e:
            print(f"⚠️ Memory optimization failed: {e}")
    
    def optimize_batch_processing(self, 
                                items: List[any], 
                                process_func: Callable,
                                batch_size: Optional[int] = None) -> List[any]:
        """
        Optimize batch processing for RTX 3050
        
        Args:
            items: List of items to process
            process_func: Function to process each item
            batch_size: Optimal batch size (auto-calculated if None)
            
        Returns:
            List of processed results
        """
        if not batch_size:
            # Calculate optimal batch size based on memory and item count
            batch_size = self._calculate_optimal_batch_size(len(items))
        
        print(f"🔄 Processing {len(items)} items in batches of {batch_size}")
        
        results = []
        
        # Process in memory-efficient batches
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            # Process batch with memory cleanup
            batch_results = []
            for item in batch:
                try:
                    result = process_func(item)
                    batch_results.append(result)
                except Exception as e:
                    print(f"⚠️ Error processing item: {e}")
                    batch_results.append(None)
            
            results.extend(batch_results)
            
            # Memory cleanup after each batch
            self._cleanup_memory()
            
            # Progress indication
            progress = min(i + batch_size, len(items))
            print(f" Progress: {progress}/{len(items)} ({progress/len(items)*100:.1f}%)")
        
        return results
    
    def _calculate_optimal_batch_size(self, total_items: int) -> int:
        """Calculate optimal batch size for RTX 3050"""
        if total_items <= 3:
            return 1  # Process individually for small batches
        elif total_items <= 10:
            return 2  # Small batches
        else:
            return 3  # Conservative batch size for 6GB VRAM
    
    def _cleanup_memory(self):
        """Aggressive memory cleanup for RTX 3050"""
        # Python garbage collection
        gc.collect()
        
        # GPU memory cleanup if available
        if self.gpu_available:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except ImportError:
                pass
    
    def parallel_cpu_processing(self, 
                              items: List[any], 
                              process_func: Callable,
                              use_threading: bool = True) -> List[any]:
        """
        Optimized CPU parallel processing
        
        Args:
            items: Items to process
            process_func: Processing function
            use_threading: Use threading vs multiprocessing
            
        Returns:
            Processed results
        """
        if len(items) <= 1:
            return [process_func(item) for item in items]
        
        results = [None] * len(items)
        
        if use_threading:
            # Threading for I/O bound tasks
            return self._thread_processing(items, process_func, results)
        else:
            # Multiprocessing for CPU bound tasks
            return self._multiprocess_processing(items, process_func)
    
    def _thread_processing(self, items: List[any], process_func: Callable, results: List) -> List:
        """Thread-based parallel processing"""
        def worker(queue: Queue, result_list: List):
            while True:
                item = queue.get()
                if item is None:
                    break
                    
                index, data = item
                try:
                    result = process_func(data)
                    result_list[index] = result
                except Exception as e:
                    print(f"Thread error: {e}")
                    result_list[index] = None
                finally:
                    queue.task_done()
        
        # Create queue and threads
        queue = Queue()
        threads = []
        
        # Start worker threads
        for _ in range(self.max_workers):
            thread = threading.Thread(target=worker, args=(queue, results))
            thread.daemon = True
            thread.start()
            threads.append(thread)
        
        # Add items to queue
        for i, item in enumerate(items):
            queue.put((i, item))
        
        # Wait for completion
        queue.join()
        
        # Stop threads
        for _ in range(self.max_workers):
            queue.put(None)
        
        for thread in threads:
            thread.join()
        
        return results
    
    def _multiprocess_processing(self, items: List[any], process_func: Callable) -> List:
        """Multiprocess-based parallel processing"""
        import concurrent.futures
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(process_func, items))
        
        return results
    
    def monitor_performance(self, func: Callable) -> Callable:
        """
        Performance monitoring decorator
        
        Args:
            func: Function to monitor
            
        Returns:
            Wrapped function with performance monitoring
        """
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            try:
                result = func(*args, **kwargs)
                success = True
            except Exception as e:
                print(f"⚠️ Function error: {e}")
                result = None
                success = False
            
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            # Performance report
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            
            status = "" if success else ""
            print(f"{status} {func.__name__}: {duration:.2f}s, Memory: {memory_delta:+.1f}MB")
            
            return result
        
        return wrapper
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return 0.0
    
    def get_optimization_report(self) -> Dict:
        """Get current optimization status"""
        return {
            'gpu_available': self.gpu_available,
            'memory_limit_mb': self.memory_limit,
            'max_workers': self.max_workers,
            'current_memory_mb': self._get_memory_usage(),
            'optimizations_active': [
                'Memory cleanup',
                'Batch processing',
                'Parallel CPU processing',
                'GPU memory management' if self.gpu_available else 'CPU optimization'
            ]
        }

# Global optimizer instance
rtx_optimizer = RTX3050Optimizer()
