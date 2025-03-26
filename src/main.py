#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RoboRadAssist - Hệ Thống Robot Hỗ Trợ Xạ Trị Phẫu Thuật
Main application entry point
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Thêm thư mục gốc vào PYTHONPATH
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

# Import các module của hệ thống
from src.core.robot_control import robot_manager
from src.core.image_processing import image_processor
from src.core.treatment_planning import treatment_planner
from src.core.dose_calculation import dose_calculator
from src.ui.desktop import desktop_app
from src.ui.web import web_server
from src.database import db_manager

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(ROOT_DIR / "logs/roboradassist.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='RoboRadAssist - Hệ Thống Robot Hỗ Trợ Xạ Trị Phẫu Thuật')
    parser.add_argument('--web', action='store_true', help='Start web interface')
    parser.add_argument('--desktop', action='store_true', help='Start desktop interface')
    parser.add_argument('--port', type=int, default=8080, help='Port for web interface')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to configuration file')
    return parser.parse_args()

def initialize_system(config_path):
    """Initialize system components."""
    logger.info("Initializing RoboRadAssist system...")
    
    # Ensure required directories exist
    os.makedirs(ROOT_DIR / "logs", exist_ok=True)
    
    # Initialize database
    db_manager.initialize()
    
    # Initialize core components
    robot_manager.initialize()
    image_processor.initialize()
    treatment_planner.initialize()
    dose_calculator.initialize()
    
    logger.info("System initialization complete")

def main():
    """Main application entry point."""
    args = parse_arguments()
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    # Initialize system
    initialize_system(args.config)
    
    # Start requested interface
    if args.web:
        logger.info(f"Starting web interface on port {args.port}")
        web_server.start(port=args.port)
    elif args.desktop:
        logger.info("Starting desktop interface")
        desktop_app.start()
    else:
        # Default to web interface
        logger.info(f"Starting web interface on port {args.port}")
        web_server.start(port=args.port)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        sys.exit(1)
