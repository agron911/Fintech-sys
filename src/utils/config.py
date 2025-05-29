import json
from pathlib import Path
import os

def load_config():
    print('Current working directory:', os.getcwd())
    config_path = Path(__file__).resolve().parent.parent.parent / 'config' / 'config.json'
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)
