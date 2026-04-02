# Scan Cache Feature - Never Re-scan Again! ⚡

## Overview
The scan cache automatically saves your Elliott Wave scan results so you don't need to re-run scans every time you restart the program. Just click "Load Last Scan" and your results appear instantly!

---

## Why This Matters 🎯

### Before (Without Cache):
```
1. Start program
2. Select timeframe
3. Click "Scan All Stocks"
4. Wait 5-10 minutes for 1000 stocks to scan
5. Review results
6. Close program

NEXT DAY:
1. Start program
2. Click "Scan All Stocks" AGAIN
3. Wait 5-10 minutes AGAIN  😩
4. Review same results...
```

### After (With Cache): ✅
```
1. Start program
2. See "LAST SCAN AVAILABLE (2h ago)"
3. Click "Load Last Scan"
4. Results appear in 1 second! ⚡
5. Start analyzing immediately
```

**Time saved: 99%** (5-10 minutes → 1 second)

---

## Features

### 1. **Automatic Saving**
- Every scan automatically saves to `cache/elliott_wave_scan_cache.json`
- Saves:
  - All stocks with patterns
  - Confidence scores
  - Pattern types
  - Timeframe used
  - Timestamp
  - Total stocks scanned

### 2. **Auto-Load Notification on Startup**
When you start the program, if a cache exists:
```
======================================================================
LAST SCAN AVAILABLE
======================================================================
Timestamp: 2h ago
Timeframe: week
Patterns found: 25/100

Click 'Load Last Scan' to restore results
======================================================================
```

### 3. **One-Click Restore**
- Click "Load Last Scan" button
- All results populate instantly
- Timeframe automatically restored
- Click any stock to view immediately

### 4. **Cache Metadata**
Cache stores:
- **Timestamp**: When scan was performed
- **Timeframe**: day/week/month
- **Chart Type**: Which chart type was used
- **Total Scanned**: How many stocks were scanned
- **Patterns Found**: How many had patterns
- **Stock Details**: Symbol, confidence, pattern group for each

---

## How It Works

### Scan Flow (with auto-save):
```
User clicks "Scan All Stocks"
    ↓
Scan runs (5-10 minutes)
    ↓
Results displayed in list
    ↓
[AUTO-SAVE] Results saved to cache/elliott_wave_scan_cache.json ✓
    ↓
"✓ Scan results saved to cache" message shown
```

### Load Flow (instant):
```
User clicks "Load Last Scan"
    ↓
Read cache/elliott_wave_scan_cache.json
    ↓
Restore timeframe setting
    ↓
Populate list with stocks
    ↓
"✓ Loaded 25 stocks from cache" message shown
    ↓
Ready to use immediately!
```

---

## Cache File Location

**Path**: `cache/elliott_wave_scan_cache.json`

**Structure**:
```json
{
  "timestamp": "2025-11-17T15:30:45.123456",
  "timeframe": "week",
  "chart_type": "Candlestick (Week)",
  "total_scanned": 100,
  "patterns_found": 25,
  "stocks": [
    {
      "symbol": "AAPL",
      "confidence": 0.15,
      "pattern_group": "primary",
      "patterns_count": 1
    },
    {
      "symbol": "MSFT",
      "confidence": 0.12,
      "pattern_group": "primary",
      "patterns_count": 1
    }
    // ... more stocks
  ]
}
```

---

## Usage Examples

### Example 1: First Scan
```
Day 1, 9:00 AM:
  User: Selects "Candlestick (Week)"
  User: Clicks "Scan All Stocks"
  System: Scans 100 stocks (takes 2 minutes)
  System: Shows "✓ Scan results saved to cache"
  System: Displays 25 stocks in list
  User: Analyzes stocks, closes program

Day 1, 3:00 PM (same day):
  User: Starts program
  System: Shows "LAST SCAN AVAILABLE (6h ago)"
  User: Clicks "Load Last Scan"
  System: Loads 25 stocks instantly ⚡
  User: Continues analysis
```

### Example 2: Different Timeframes
```
User: Scans with "Week" → 25 patterns found, cached
User: Changes to "Day", clicks "Scan All Stocks"
      → 30 patterns found, cache UPDATED (overwrites)
User: Closes program

Next time:
  System: Shows last cache (Day timeframe, 30 patterns)
  User: Loads it, sees Day patterns

User: Wants weekly patterns again?
  → Just change to "Week" and scan again
  → New cache created
```

---

## Technical Details

### Files Created/Modified

**New Files:**
1. `src/utils/scan_cache.py` - Cache management system
   - `ScanCache` class
   - `save_scan_results()`
   - `load_scan_results()`
   - `get_cache_info()`
   - `is_cache_fresh()`
   - `clear_cache()`

**Modified Files:**
1. `gui/frame.py`
   - Added `load_scan_button` (line 85-86)
   - Added `scan_cache` initialization (line 116-118)
   - Added `auto_load_last_scan()` method (line 246-261)
   - Added `safe_handle_load_scan()` handler (line 238-244)

2. `gui/handlers.py`
   - Added cache save after scan (line 363-375)
   - Added `handle_load_scan()` function (line 416-467)
   - Updated exports (line 1024)

### Cache Size
- Typical cache file: **5-20 KB** for 100 stocks
- Negligible disk space usage
- Fast to read/write

### Cache Expiration
- **No automatic expiration** - cache persists until overwritten
- Check cache age with `get_cache_info()['age_seconds']`
- Recommended: Re-scan weekly for fresh data

---

## Cache Management

### Manually Clear Cache

**Option 1: Delete file**
```bash
rm cache/elliott_wave_scan_cache.json
```

**Option 2: Programmatic (future feature)**
```python
# Could add "Clear Cache" button
self.scan_cache.clear_cache()
```

### Force Fresh Scan
Just click "Scan All Stocks" - it will overwrite the cache with new results.

### Check Cache Status
Cache metadata includes:
- Age (e.g., "2h ago", "1d ago")
- Timestamp (ISO format)
- Number of patterns
- Timeframe used

---

## Benefits Summary

✅ **Save Time**: 99% faster than re-scanning (5-10 min → 1 sec)
✅ **Persist Results**: Scan once, use forever
✅ **Auto-Save**: No manual export/import needed
✅ **Auto-Notify**: Shows cache status on startup
✅ **One-Click Load**: Single button to restore
✅ **Timeframe Preserved**: Remembers which timeframe was scanned
✅ **Small Size**: Only 5-20 KB per cache file
✅ **No Configuration**: Works automatically

---

## Troubleshooting

### Problem: Cache not loading
**Solutions**:
1. Check if `cache/elliott_wave_scan_cache.json` exists
2. Check file permissions (read access needed)
3. Verify JSON is valid (not corrupted)
4. Check output window for error messages

### Problem: "No cached scan results found"
**Cause**: You haven't run a scan yet, or cache file was deleted
**Solution**: Click "Scan All Stocks" to create cache

### Problem: Cache is outdated
**Solution**:
- Just run "Scan All Stocks" again
- New cache automatically replaces old one

### Problem: Want different timeframe
**Solution**:
- Cache only stores ONE scan at a time
- Change chart type and scan again
- New scan overwrites old cache

---

## Future Enhancements (Not Yet Implemented)

- [ ] Multiple caches (one per timeframe)
- [ ] Cache age warning (e.g., "Cache is 7 days old")
- [ ] "Clear Cache" button
- [ ] Export cache to CSV
- [ ] Import/share cache with others
- [ ] Cache compression for large datasets
- [ ] Auto-refresh cache on schedule
- [ ] Cache for individual stocks (not just scan results)

---

## Comparison: Before vs After

| Feature | Before (No Cache) | After (With Cache) |
|---------|------------------|-------------------|
| Re-scan time | 5-10 minutes | 1 second ⚡ |
| Startup time | Wait for manual scan | Auto-notify + instant load |
| Data persistence | Lost on close | Saved automatically |
| Workflow | Scan → Analyze → Close → Scan again | Scan → Analyze → Close → Load → Continue |
| User action | Manual scan every time | One-click load |
| Disk space | 0 KB | 5-20 KB |

---

## Related Documentation

- [SCAN_FEATURE.md](SCAN_FEATURE.md) - Main scan feature documentation
- [OPTIMIZATION_CHANGES.md](OPTIMIZATION_CHANGES.md) - Pattern detection improvements
- [GUI_LAYOUT.txt](GUI_LAYOUT.txt) - Visual layout guide
