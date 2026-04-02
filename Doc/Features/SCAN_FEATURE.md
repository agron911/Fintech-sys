# Elliott Wave Pattern Scanner Feature

## Overview
A new "Scan All Stocks" button that automatically scans your entire stock database for Elliott Wave patterns and displays results in an interactive list.

## Features

### 1. **Scan All Stocks Button**
- Located in the main button row (next to other analysis buttons)
- Scans all stocks in your data directory
- Uses the currently selected **Chart Type** (Day/Week/Month/Line)
- Runs in background thread - doesn't freeze the GUI
- **Automatically saves results to cache** for later use

### 1.5. **Load Last Scan Button** ⭐ NEW
- Instantly loads previously saved scan results
- **No need to re-scan when you restart the program**
- Shows cache age and scan details
- Restores the exact timeframe that was scanned
- **Auto-loads on startup** - shows cache info automatically

### 2. **Interactive Results List**
- Shows on the left side of the chart area
- Displays stocks sorted by confidence (highest first)
- Format: `SYMBOL (confidence%) icon`
  - 🟢 = Primary pattern (high confidence)
  - 🟡 = Alternative pattern (moderate confidence)

### 3. **Click to View**
- Click any stock in the list to instantly view its Elliott Wave pattern
- Automatically updates the chart with the selected stock
- **Important**: Uses the SAME timeframe that was used during the scan
  - If you scanned with "Week", clicking a stock will show weekly patterns
  - Chart type dropdown automatically updates to match
  - This ensures you see the pattern that was detected

### 4. **Progress Feedback**
- Real-time progress in the output window
- Shows: `[current/total] status SYMBOL: confidence (group)`
- Status icons:
  - ✓ = Pattern found
  - ✗ = No pattern or error
  - ~ = Alternative pattern

### 5. **Summary Report**
- Shows total stocks scanned
- Lists stocks with patterns found
- Displays top 10 patterns by confidence

---

## How to Use

### Basic Workflow

**First Time (Scan):**

1. **Select Chart Type** (optional)
   - Choose Day/Week/Month/Line from dropdown
   - This determines which timeframe to analyze

2. **Click "Scan All Stocks"**
   - Button starts the scan
   - All buttons disabled during scan
   - Progress shown in output window
   - **Results automatically saved to cache**

3. **Review Results**
   - Left panel shows all stocks with patterns
   - Sorted by confidence (best first)
   - Click any stock to view its chart

4. **Analyze Individual Stocks**
   - Click stock in list → Chart updates automatically
   - Use "Show Elliott Wave" for detailed multi-pattern view
   - Change chart type and re-scan for different timeframes

---

**Next Time (Load Cache):** ⭐ NEW

1. **Start the program**
   - Output window shows: "LAST SCAN AVAILABLE"
   - Shows when scan was done and how many patterns found

2. **Click "Load Last Scan"**
   - Instantly restores previous scan results
   - No waiting for re-scan!
   - List populates with all stocks

3. **Click stocks to view**
   - Same as before - click any stock from list
   - Charts load immediately

### Example Session

```
1. Select "Candlestick (Week)" chart type
2. Click "Scan All Stocks"
3. Wait for scan to complete (~30 seconds for 100 stocks)
4. Review list: "AAPL (15%) 🟢", "MSFT (12%) 🟢", "TSLA (8%) 🟡"...
5. Click "AAPL (15%) 🟢" to view its weekly pattern
6. Switch to "Candlestick (Day)" and re-scan for daily patterns
```

---

## Performance Optimization

The scan is optimized for speed:
- **Fast mode**: Only finds best pattern per stock (not all alternatives)
- **Skip relationships**: Doesn't analyze cross-timeframe relationships
- **Single timeframe**: Only analyzes the selected chart type
- **Parallel processing**: Continues even if some stocks error out

### Scan Speed Estimates
- 100 stocks: ~30-60 seconds
- 500 stocks: ~2-5 minutes
- 1000 stocks: ~5-10 minutes

---

## Technical Details

### Files Modified

1. **gui/frame.py**
   - Added `scan_button` (line 82-83)
   - Added `stocks_list` ListBox widget (line 91-93)
   - Added `on_stock_selected()` handler (line 226-239)
   - Added `safe_handle_scan_all_stocks()` handler (line 214-224)
   - Updated `enable_buttons()` to include scan button (line 178)

2. **gui/handlers.py**
   - Added `handle_scan_all_stocks()` function (line 274-384)
   - Added `_update_stocks_listbox()` helper (line 387-396)
   - Updated `__all__` exports (line 953)

### Function Flow

```
User clicks "Scan All Stocks"
    ↓
safe_handle_scan_all_stocks() [frame.py:214]
    ↓
handle_scan_all_stocks() [handlers.py:274]
    ↓
For each stock:
    - load_and_preprocess_data()
    - find_elliott_wave_patterns_advanced()
    - Check confidence threshold (≥5%)
    ↓
_update_stocks_listbox() [handlers.py:387]
    ↓
User clicks stock in list
    ↓
on_stock_selected() [frame.py:226]
    ↓
handle_show_elliott_wave() [handlers.py:183]
```

### Configuration

**Minimum Confidence Threshold**: 5% (line 334 in handlers.py)
- Adjust this to filter more/fewer results
- Lower = More patterns (may include false positives)
- Higher = Fewer patterns (only high confidence)

**Chart Type Mapping** (line 294-300 in handlers.py):
```python
chart_type_map = {
    'Candlestick (Day)': 'day',
    'Candlestick (Week)': 'week',
    'Candlestick (Month)': 'month',
    'Line': 'day'
}
```

---

## Troubleshooting

### Problem: Scan button disabled
**Solution**: Another operation is running. Wait for it to complete.

### Problem: No stocks in list after scan
**Possible causes**:
1. No patterns found (try lowering thresholds in OPTIMIZATION_CHANGES.md)
2. Insufficient data (stocks need ≥50 data points)
3. Wrong data directory (check config)

### Problem: Scan is very slow
**Solutions**:
1. Reduce number of stocks (filter by exchange)
2. Use faster chart type (Monthly < Weekly < Daily)
3. Ensure SSD for data directory
4. Close other applications

### Problem: "Error selecting stock"
**Solution**:
1. Check stock data file exists
2. Verify stock has sufficient data
3. Check output window for detailed error

### Problem: Clicked stock shows "No pattern found" even though it's in the list
**This was FIXED in the latest version!**
- **Root cause**: Scan used one timeframe (e.g., "Week"), but clicking analyzed a different timeframe (e.g., "Day")
- **Solution**: System now automatically uses the SAME timeframe that was scanned
- **What you'll see**: Output window shows "Displaying SYMBOL using scanned timeframe: week"
- **Note**: The chart type dropdown will automatically change to match the scanned timeframe

---

## Tips for Best Results

1. **Weekly charts** often work best for Elliott Wave patterns (clearer trends)
2. **Scan different timeframes** - patterns visible on weekly may not show on daily
3. **Focus on high confidence** (>10%) patterns first
4. **Re-scan periodically** as new data arrives
5. **Use with other indicators** - Elliott Wave is one tool among many

---

## Future Enhancements (Not Yet Implemented)

- [ ] Save/load scan results
- [ ] Export pattern list to CSV
- [ ] Filter by confidence range
- [ ] Parallel scanning (multi-threading)
- [ ] Progress bar instead of text output
- [ ] Pattern type filter (impulse vs. corrective)
- [ ] Scan specific exchanges only
- [ ] Email alerts for new patterns

---

## Related Documentation

- [OPTIMIZATION_CHANGES.md](OPTIMIZATION_CHANGES.md) - Threshold adjustments
- [STRATEGY_QUICKSTART.md](STRATEGY_QUICKSTART.md) - Using patterns for trading
- [PERFORMANCE_OPTIMIZATION_GUIDE.md](PERFORMANCE_OPTIMIZATION_GUIDE.md) - Speed tips
