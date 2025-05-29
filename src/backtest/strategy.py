def backtest_elliott_strategy(df, wave_points, buy_on=0, sell_on=5, column='Close'):
    """
    Simulate buying at wave_points[buy_on] and selling at wave_points[sell_on].
    Returns profit and trade details.
    """
    if len(wave_points) < max(buy_on, sell_on) + 1:
        return None, None
    buy_idx = wave_points[buy_on]
    sell_idx = wave_points[sell_on]
    buy_price = df[column].iloc[buy_idx]
    sell_price = df[column].iloc[sell_idx]
    profit = sell_price - buy_price
    return profit, {'buy_idx': buy_idx, 'sell_idx': sell_idx, 'buy_price': buy_price, 'sell_price': sell_price} 