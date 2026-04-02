import json
from pathlib import Path
import os
import logging

logger = logging.getLogger(__name__)


def load_config():
    logger.info(f'Current working directory: {os.getcwd()}')
    config_path = Path(__file__).resolve().parent.parent.parent / 'config' / 'config.json'
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)
