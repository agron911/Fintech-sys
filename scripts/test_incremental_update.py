"""
Test script for incremental data update functionality.
Tests the should_update_file() logic and verifies that old/new files are handled correctly.
"""

import sys
from pathlib import Path
import datetime as dt
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.crawler.yahoo_finance import should_update_file, UPDATE_INTERVAL_DAYS


def test_should_update_file():
    """Test the should_update_file logic."""
    
    test_dir = Path("data/raw/test_incremental")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Test 1: Non-existent file should return True (needs update)
        logger.info("\n=== TEST 1: Non-existent file ===")
        non_existent = test_dir / "non_existent.txt"
        result = should_update_file(non_existent)
        assert result == True, "Non-existent file should need update"
        logger.info(f"✓ Non-existent file returns: {result} (expected: True)")
        
        # Test 2: New file (< UPDATE_INTERVAL_DAYS old) should return False
        logger.info("\n=== TEST 2: Recent file (newer than UPDATE_INTERVAL_DAYS) ===")
        new_file = test_dir / "new_file.txt"
        new_file.write_text("test data")
        result = should_update_file(new_file)
        assert result == False, "Recent file should NOT need update"
        logger.info(f"✓ Recent file (0 days old) returns: {result} (expected: False)")
        
        # Test 3: Old file (>= UPDATE_INTERVAL_DAYS) should return True
        logger.info(f"\n=== TEST 3: Old file (>= {UPDATE_INTERVAL_DAYS} days old) ===")
        old_file = test_dir / "old_file.txt"
        old_file.write_text("test data")
        
        # Modify file timestamp to be 10 days old
        old_timestamp = time.time() - (10 * 24 * 60 * 60)
        Path(old_file).touch()
        import os
        os.utime(old_file, (old_timestamp, old_timestamp))
        
        result = should_update_file(old_file)
        assert result == True, "Old file should need update"
        
        file_age = (dt.datetime.now() - dt.datetime.fromtimestamp(old_file.stat().st_mtime)).days
        logger.info(f"✓ Old file ({file_age} days old) returns: {result} (expected: True)")
        
        # Test 4: Edge case - file exactly UPDATE_INTERVAL_DAYS old
        logger.info(f"\n=== TEST 4: Edge case file (exactly {UPDATE_INTERVAL_DAYS} days old) ===")
        edge_file = test_dir / "edge_file.txt"
        edge_file.write_text("test data")
        
        edge_timestamp = time.time() - (UPDATE_INTERVAL_DAYS * 24 * 60 * 60)
        os.utime(edge_file, (edge_timestamp, edge_timestamp))
        
        result = should_update_file(edge_file)
        assert result == True, f"File exactly {UPDATE_INTERVAL_DAYS} days old should need update"
        
        file_age = (dt.datetime.now() - dt.datetime.fromtimestamp(edge_file.stat().st_mtime)).days
        logger.info(f"✓ Edge file ({file_age} days old) returns: {result} (expected: True)")
        
        logger.info("\n" + "="*50)
        logger.info("✅ ALL TESTS PASSED!")
        logger.info("="*50)
        
    finally:
        # Cleanup
        import shutil
        if test_dir.exists():
            shutil.rmtree(test_dir)
            logger.info(f"\nCleaned up test directory: {test_dir}")


if __name__ == "__main__":
    test_should_update_file()
