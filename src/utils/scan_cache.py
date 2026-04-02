"""
Caching system for Elliott Wave scan results.
Allows saving and loading scan results to avoid re-scanning all stocks.
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)



class ScanCache:
    """Manages caching of Elliott Wave scan results"""

    def __init__(self, cache_dir: str = "cache"):
        """
        Initialize scan cache

        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "elliott_wave_scan_cache.json"

    def save_scan_results(self,
                         stocks_with_patterns: List[Dict[str, Any]],
                         timeframe: str,
                         chart_type: str,
                         total_scanned: int) -> bool:
        """
        Save scan results to cache file

        Args:
            stocks_with_patterns: List of stocks with detected patterns
            timeframe: Timeframe used for scan ('day', 'week', 'month')
            chart_type: Chart type used for scan
            total_scanned: Total number of stocks scanned

        Returns:
            True if save successful, False otherwise
        """
        try:
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'timeframe': timeframe,
                'chart_type': chart_type,
                'total_scanned': total_scanned,
                'patterns_found': len(stocks_with_patterns),
                'stocks': stocks_with_patterns
            }

            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)

            return True
        except Exception as e:
            logger.info(f"Error saving scan cache: {e}")
            return False

    def load_scan_results(self) -> Optional[Dict[str, Any]]:
        """
        Load scan results from cache file

        Returns:
            Dictionary with cache data or None if cache doesn't exist/is invalid
        """
        try:
            if not self.cache_file.exists():
                return None

            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)

            # Validate cache structure
            required_keys = ['timestamp', 'timeframe', 'chart_type', 'stocks']
            if not all(key in cache_data for key in required_keys):
                return None

            return cache_data
        except Exception as e:
            logger.info(f"Error loading scan cache: {e}")
            return None

    def get_cache_info(self) -> Optional[Dict[str, Any]]:
        """
        Get cache metadata without loading full results

        Returns:
            Dictionary with cache metadata or None if no cache exists
        """
        try:
            if not self.cache_file.exists():
                return None

            cache_data = self.load_scan_results()
            if not cache_data:
                return None

            # Parse timestamp
            timestamp = datetime.fromisoformat(cache_data['timestamp'])
            age_seconds = (datetime.now() - timestamp).total_seconds()

            return {
                'timestamp': cache_data['timestamp'],
                'age_seconds': age_seconds,
                'age_formatted': self._format_age(age_seconds),
                'timeframe': cache_data['timeframe'],
                'chart_type': cache_data['chart_type'],
                'total_scanned': cache_data.get('total_scanned', 0),
                'patterns_found': cache_data.get('patterns_found', len(cache_data['stocks'])),
                'exists': True
            }
        except Exception as e:
            logger.info(f"Error getting cache info: {e}")
            return None

    def clear_cache(self) -> bool:
        """
        Delete the cache file

        Returns:
            True if successful, False otherwise
        """
        try:
            if self.cache_file.exists():
                self.cache_file.unlink()
            return True
        except Exception as e:
            logger.info(f"Error clearing cache: {e}")
            return False

    @staticmethod
    def _format_age(age_seconds: float) -> str:
        """Format cache age in human-readable form"""
        if age_seconds < 60:
            return f"{int(age_seconds)}s ago"
        elif age_seconds < 3600:
            return f"{int(age_seconds / 60)}m ago"
        elif age_seconds < 86400:
            return f"{int(age_seconds / 3600)}h ago"
        else:
            return f"{int(age_seconds / 86400)}d ago"

    def is_cache_fresh(self, max_age_hours: int = 24) -> bool:
        """
        Check if cache is fresh (within max age)

        Args:
            max_age_hours: Maximum age in hours to consider cache fresh

        Returns:
            True if cache exists and is fresh, False otherwise
        """
        info = self.get_cache_info()
        if not info:
            return False

        max_age_seconds = max_age_hours * 3600
        return info['age_seconds'] < max_age_seconds
