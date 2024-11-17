"""Helper utilities for the Agent Reasoning Beta platform."""

from __future__ import annotations

import time
import asyncio
import functools
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from datetime import datetime, timezone

from utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Callable:
    """Retry decorator with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts.
        delay: Initial delay between retries in seconds.
        backoff: Multiplier for delay after each retry.
        exceptions: Tuple of exceptions to catch and retry.
    
    Returns:
        Decorated function.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed: {str(e)}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_attempts} attempts failed")
                        raise
            
            raise last_exception  # Should never reach here
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed: {str(e)}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_attempts} attempts failed")
                        raise
            
            raise last_exception  # Should never reach here
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

def timed(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to measure and log function execution time.
    
    Args:
        func: Function to time.
    
    Returns:
        Decorated function.
    """
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs) -> T:
        start_time = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            elapsed = time.perf_counter() - start_time
            logger.debug(f"{func.__name__} completed in {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            logger.error(f"{func.__name__} failed after {elapsed:.3f}s: {str(e)}")
            raise
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs) -> T:
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start_time
            logger.debug(f"{func.__name__} completed in {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            logger.error(f"{func.__name__} failed after {elapsed:.3f}s: {str(e)}")
            raise
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to specified length.
    
    Args:
        text: Text to truncate.
        max_length: Maximum length of truncated text.
        suffix: String to append to truncated text.
    
    Returns:
        Truncated text.
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def format_timestamp(
    timestamp: Optional[Union[int, float, datetime]] = None,
    format: str = "%Y-%m-%d %H:%M:%S UTC"
) -> str:
    """Format timestamp as string.
    
    Args:
        timestamp: Unix timestamp or datetime object. If None, uses current time.
        format: strftime format string.
    
    Returns:
        Formatted timestamp string.
    """
    if timestamp is None:
        dt = datetime.now(timezone.utc)
    elif isinstance(timestamp, (int, float)):
        dt = datetime.fromtimestamp(timestamp, timezone.utc)
    else:
        dt = timestamp.astimezone(timezone.utc)
    
    return dt.strftime(format)

def batch_items(items: List[T], batch_size: int) -> List[List[T]]:
    """Split list into batches.
    
    Args:
        items: List to split.
        batch_size: Size of each batch.
    
    Returns:
        List of batches.
    """
    return [
        items[i:i + batch_size]
        for i in range(0, len(items), batch_size)
    ]

def deep_update(base: Dict, update: Dict) -> Dict:
    """Recursively update a dictionary.
    
    Args:
        base: Base dictionary to update.
        update: Dictionary with updates.
    
    Returns:
        Updated dictionary.
    """
    for key, value in update.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base

def safe_divide(
    numerator: Union[int, float],
    denominator: Union[int, float],
    default: Union[int, float] = 0
) -> Union[int, float]:
    """Safely divide two numbers.
    
    Args:
        numerator: Number to divide.
        denominator: Number to divide by.
        default: Default value to return if denominator is zero.
    
    Returns:
        Result of division or default value.
    """
    try:
        return numerator / denominator
    except ZeroDivisionError:
        return default
