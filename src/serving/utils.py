import socket
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from huggingface_hub import HfApi
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.warning("huggingface_hub not available. Install with: pip install huggingface_hub")

def validate_huggingface_model(model_name):
    if not HF_AVAILABLE:
        logger.warning(f"Cannot validate HuggingFace model '{model_name}' - huggingface_hub not installed")
        return True
    
    try:
        api = HfApi()
        api.model_info(model_name)
        logger.info(f"Validated HuggingFace model: {model_name}")
        return True
    except Exception as e:
        logger.error(f"Invalid HuggingFace model '{model_name}': {str(e)}")
        return False

def log_stream(stream, log_file, prefix):
    with open(log_file, 'a') as f:
        while True:
            line = stream.readline()
            if not line:
                break
            logger.info(f"[{prefix}] {line.strip()}")
            f.write(line)
            f.flush()

def check_port_available(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(('localhost', port))
        return True
    except socket.error:
        return False
    finally:
        sock.close()

def get_last_log_lines(log_file, n=20):
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            return ''.join(lines[-n:]) if lines else ''
    except Exception as e:
        return f"Could not read log file: {str(e)}" 