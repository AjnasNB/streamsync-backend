import sys
import os
import logging
import subprocess
import platform
import json
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def check_python_dependencies():
    """Check if all required Python dependencies are installed."""
    try:
        import flask
        import flask_cors
        import ffmpeg
        import uuid
        import requests
        logger.info("All Python dependencies are installed.")
        return True
    except ImportError as e:
        logger.error(f"Missing Python dependency: {e}")
        return False

def check_ffmpeg_installed():
    """Check if FFmpeg is installed on the system."""
    try:
        if platform.system() == "Windows":
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        else:
            subprocess.run(["which", "ffmpeg"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        logger.info("FFmpeg is installed.")
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.error("FFmpeg is not installed or not in PATH.")
        return False

def setup_environment():
    """Load environment variables and set up the environment."""
    # Determine the root directory of the project
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Load environment variables from .env file if it exists
    env_file = os.path.join(project_root, '.env')
    if os.path.exists(env_file):
        load_dotenv(env_file)
        logger.info(f"Loading environment variables from {env_file}")
    
    # Create videos directory if it doesn't exist
    videos_dir = os.path.join(project_root, 'videos')
    if not os.path.exists(videos_dir):
        os.makedirs(videos_dir)
        logger.info(f"Created videos directory at {videos_dir}")
    
    # Create data directory if it doesn't exist
    data_dir = os.path.join(project_root, 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        logger.info(f"Created data directory at {data_dir}")
    
    # Create user data file if it doesn't exist
    users_file = os.path.join(data_dir, 'users.json')
    if not os.path.exists(users_file):
        with open(users_file, 'w') as f:
            json.dump({}, f)
        logger.info(f"Created empty users data file at {users_file}")
    
    # Create videos data file if it doesn't exist
    videos_file = os.path.join(data_dir, 'videos.json')
    if not os.path.exists(videos_file):
        with open(videos_file, 'w') as f:
            json.dump({}, f)
        logger.info(f"Created empty videos data file at {videos_file}")
    
    # Create stream status file if it doesn't exist
    stream_file = os.path.join(data_dir, 'stream_status.json')
    if not os.path.exists(stream_file):
        with open(stream_file, 'w') as f:
            json.dump({"running": False, "start_time": None}, f)
        logger.info(f"Created stream status file at {stream_file}")
    
    logger.info("Environment setup complete.")

def start_server():
    """Start the Flask server."""
    from app import app
    
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'True').lower() == 'true'
    
    logger.info(f"Starting server on {host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    logger.info("Checking system configuration...")
    
    if not check_python_dependencies() or not check_ffmpeg_installed():
        logger.error("Failed system configuration check. Exiting.")
        sys.exit(1)
    
    setup_environment()
    
    logger.info("Starting StreamSync backend server...")
    start_server() 