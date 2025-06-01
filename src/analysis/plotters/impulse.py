import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_impulse(df: pd.DataFrame, points: dict, config: dict = None, ax=None, title: str = 'Impulse Wave Analysis'):
    """
    Plot impulse wave analysis using matplotlib. Points dict should contain keys like 'impulse_wave', 'corrective_wave', 'peaks', 'troughs'.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 10))
    else:
        fig = ax.figure
    column = config.get('column', 'close') if config else 'close'
    ax.clear()
    ax.plot(df.index, df[column], label='Price', alpha=0.8, color='black', linewidth=1.5)
    if 'peaks' in points and len(points['peaks']) > 0:
        ax.scatter(df.index[points['peaks']], df[column].iloc[points['peaks']], c='red', marker='^', s=50, alpha=0.7, label='Peaks', zorder=5)
    if 'troughs' in points and len(points['troughs']) > 0:
        ax.scatter(df.index[points['troughs']], df[column].iloc[points['troughs']], c='green', marker='v', s=50, alpha=0.7, label='Troughs', zorder=5)
    if 'impulse_wave' in points and len(points['impulse_wave']) > 0:
        ax.plot(df.index[points['impulse_wave']], df[column].iloc[points['impulse_wave']], 'bo-', label='Impulse Wave', markersize=10, linewidth=3, alpha=0.9, zorder=10)
    if 'corrective_wave' in points and len(points['corrective_wave']) > 0:
        ax.plot(df.index[points['corrective_wave']], df[column].iloc[points['corrective_wave']], 'mo-', label='Corrective Wave', markersize=10, linewidth=3, alpha=0.9, linestyle='--', zorder=10)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Price', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_facecolor('#fafafa')
    if len(df) > 0:
        ax.set_xlim(df.index.min(), df.index.max())
        price_range = df[column].max() - df[column].min()
        ax.set_ylim(df[column].min() - price_range * 0.05, df[column].max() + price_range * 0.1)
    fig.tight_layout()
    if ax is None:
        plt.show() 