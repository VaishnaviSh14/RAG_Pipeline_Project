import logging
import os

log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'log')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'app.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

def get_logger(name: str = None):
    """Get a configured logger instance."""
    return logging.getLogger(name)
