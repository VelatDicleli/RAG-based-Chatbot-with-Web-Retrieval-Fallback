import logging

def setup_logger(name: str = __name__) -> logging.Logger:
    """Configure and return a logger with the given name."""
    logger = logging.getLogger(name)
    
    # Configure if no handlers exist (prevents duplicate handlers)
    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    return logger