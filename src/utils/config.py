import json
from pathlib import Path
from datetime import datetime, timedelta
import os
import logging

logger = logging.getLogger(__name__)

REQUIRED_KEYS = ['data_dir', 'stk2_dir', 'start_date']


def _validate_config(config, config_path):
    """Validate config keys and date values. Mutates config in place."""
    # Check required keys
    missing = [k for k in REQUIRED_KEYS if k not in config]
    if missing:
        raise ValueError(
            f"Config file {config_path} is missing required keys: {', '.join(missing)}. "
            f"Required keys are: {', '.join(REQUIRED_KEYS)}"
        )

    # Validate / default end_date
    today = datetime.now().date()
    one_year_future = today + timedelta(days=365)

    end_date_str = config.get('end_date')
    if not end_date_str:
        config['end_date'] = today.strftime('%Y-%m-%d')
        logger.info("end_date not specified, defaulting to today (%s)", config['end_date'])
    else:
        try:
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
        except ValueError:
            logger.warning(
                "end_date '%s' is not a valid YYYY-MM-DD date, defaulting to today",
                end_date_str,
            )
            config['end_date'] = today.strftime('%Y-%m-%d')
            return config

        if end_date < today:
            logger.warning(
                "end_date '%s' is in the past, defaulting to today (%s)",
                end_date_str, today.strftime('%Y-%m-%d'),
            )
            config['end_date'] = today.strftime('%Y-%m-%d')
        elif end_date > one_year_future:
            logger.warning(
                "end_date '%s' is more than 1 year in the future", end_date_str
            )

    return config


def load_config():
    logger.info(f'Current working directory: {os.getcwd()}')
    config_path = Path(__file__).resolve().parent.parent.parent / 'config' / 'config.json'

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found at: {config_path}\n"
            f"Expected path: <project_root>/config/config.json\n"
            f"Copy config/config.example.json to config/config.json and edit as needed."
        )

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Malformed JSON in config file {config_path}: {e}\n"
            f"Please fix the JSON syntax and try again."
        ) from e

    _validate_config(config, config_path)

    # Ensure all configured directories exist (prevents errors on fresh clones)
    project_root = config_path.parent.parent
    _dir_keys = ['data_dir', 'stk2_dir', 'adjustments_dir', 'processed_dir', 'databases_dir']
    for key in _dir_keys:
        if key in config:
            dir_path = Path(config[key])
            if not dir_path.is_absolute():
                dir_path = project_root / dir_path
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug('Ensured directory exists: %s', dir_path)

    return config
