import logging
import sys
import os
from datetime import datetime

# Create logger
logger = logging.getLogger('pipeline_logger')
logger.setLevel(logging.DEBUG)

# Create file handler
log_file = os.path.join(os.path.dirname(__file__), 'logger.log')
file_handler = logging.FileHandler(log_file, mode='w')
file_handler.setLevel(logging.DEBUG)

# Create console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Log startup
logger.info("=" * 80)
logger.info("PIPELINE EXECUTION STARTED")
logger.info(f"Timestamp: {datetime.now().isoformat()}")
logger.info(f"Working directory: {os.getcwd()}")
logger.info("=" * 80)

def get_logger():
    return logger

def log_step(step_name, message):
    logger.info(f"[{step_name}] {message}")

def log_error(step_name, error):
    logger.error(f"[{step_name}] ERROR: {error}", exc_info=True)

def log_warning(step_name, message):
    logger.warning(f"[{step_name}] {message}")

def log_success(step_name, message):
    logger.info(f"[{step_name}] SUCCESS: {message}")