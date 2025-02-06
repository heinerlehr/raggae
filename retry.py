import time
from loguru import logger
from functools import wraps

def retry(retries=3, delay=1, exceptions=(Exception,)):
    """
    A decorator that retries a function if it raises specified exceptions.

    Args:
        retries (int): Number of retries before giving up.
        delay (int | float): Time to wait (in seconds) between retries.
        exceptions (tuple): Tuple of exception classes to catch and retry.
    
    Returns:
        function: Decorated function with retry logic.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            while attempt < retries:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    if attempt >= retries:
                        raise
                    logger.warning(f"Retrying due to {e}, attempt {attempt}/{retries}. Waiting {delay} seconds...")
                    time.sleep(delay)
        return wrapper
    return decorator
