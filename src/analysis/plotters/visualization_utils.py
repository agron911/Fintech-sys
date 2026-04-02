"""
Shared visualization utilities for Elliott Wave analysis to eliminate duplication.
This module consolidates common plotting functions used across multiple visualization modules.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from matplotlib.patches import FancyBboxPatch
import logging

logger = logging.getLogger(__name__)



def fix_date_labels_enhanced(ax, df: pd.DataFrame):
    """
    Enhanced date label formatting for matplotlib axes.
    
    Args:
        ax: Matplotlib axes object
        df: DataFrame with datetime index
    """
    try:
        # Set major and minor locators
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        # Rotate labels for better readability
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Auto-adjust layout to prevent label cutoff
        ax.figure.tight_layout()
        
    except Exception as e:
        logger.info(f"Warning: Could not format date labels: {e}")


def get_confidence_color(confidence: float) -> str:
    """
    Get color based on confidence level.
    
    Args:
        confidence: Confidence score (0.0 to 1.0)
    
    Returns:
        Color string
    """
    if confidence >= 0.7:
        return '#2E8B57'  # Sea Green - High confidence
    elif confidence >= 0.4:
        return '#FFA500'  # Orange - Moderate confidence
    else:
        return '#DC143C'  # Crimson - Low confidence


def get_pattern_color_scheme() -> Dict[str, str]:
    """
    Get standardized color scheme for different pattern types.
    
    Returns:
        Dictionary mapping pattern types to colors
    """
    return {
        'primary': '#2E86AB',      # Blue for primary patterns
        'supporting': '#A23B72',   # Purple for supporting patterns
        'conflicting': '#F18F01',  # Orange for conflicting patterns
        'independent': '#C73E1D',  # Red for independent patterns
        'alternative': '#8B4513',  # Brown for alternative patterns
        'impulse': '#2E86AB',      # Blue for impulse waves
        'corrective': '#E63946',   # Red for corrective waves
        'wave_line': '#F77F00'     # Orange for wave connections
    }


def add_wave_labels(ax: plt.Axes, dates: pd.DatetimeIndex, prices: pd.Series, 
                   wave_points: List[int], wave_type: str = 'impulse') -> None:
    """
    Add wave labels to the chart.
    
    Args:
        ax: Matplotlib axes
        dates: Datetime index
        prices: Price series
        wave_points: List of wave point indices
        wave_type: Type of wave ('impulse' or 'corrective')
    """
    if len(wave_points) < 2:
        return
    
    color_scheme = get_pattern_color_scheme()
    wave_color = color_scheme.get(wave_type, '#2E86AB')
    
    if wave_type == 'impulse':
        wave_labels = ['S', '1', '2', '3', '4', '5'][:len(wave_points)]
        marker_style = 'o'
    else:
        wave_labels = ['A', 'B', 'C', 'D', 'E'][:len(wave_points)]
        marker_style = 's'
    
    for i, (idx, label) in enumerate(zip(wave_points, wave_labels)):
        if 0 <= idx < len(dates) and 0 <= idx < len(prices):
            date = dates[idx]
            price = prices.iloc[idx]
            
            # Add marker
            ax.plot(date, price, marker=marker_style, markersize=8, 
                   markerfacecolor='white', markeredgecolor=wave_color, 
                   markeredgewidth=2, zorder=5)
            
            # Add label
            ax.text(date, price, label, fontsize=10, fontweight='bold',
                   color=wave_color, ha='center', va='center', zorder=6,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', 
                           edgecolor=wave_color, alpha=0.8))


def add_fibonacci_levels(ax: plt.Axes, df: pd.DataFrame, impulse_wave: np.ndarray, 
                        column: str = 'close') -> None:
    """
    Add Fibonacci retracement levels to the chart.
    
    Args:
        ax: Matplotlib axes
        df: Price DataFrame
        impulse_wave: Array of impulse wave indices
        column: Column name for price data
    """
    try:
        if len(impulse_wave) < 5:
            return
        
        valid_indices = [idx for idx in impulse_wave if 0 <= idx < len(df)]
        if len(valid_indices) < 5:
            return
        
        prices = df[column].iloc[valid_indices]
        dates = df.index[valid_indices]
        
        # Wave 1 Fibonacci retracement levels for Wave 2
        wave_1_start = prices.iloc[0]
        wave_1_end = prices.iloc[1]
        wave_1_range = wave_1_end - wave_1_start
        
        fib_retracements = [0.236, 0.382, 0.5, 0.618, 0.786]
        colors = ['#FFD700', '#FFA500', '#FF6347', '#FF4500', '#FF0000']
        
        for i, fib in enumerate(fib_retracements):
            level = wave_1_end - (wave_1_range * fib)
            ax.axhline(level, color=colors[i], linestyle=':', alpha=0.6, linewidth=1.5)
            
            if len(dates) > 2:
                text_date = dates[2]
                ax.text(text_date, level, f'{fib:.1%}', 
                       fontsize=8, color=colors[i], fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
        # Wave 3 extension levels
        wave_1_length = abs(wave_1_range)
        wave_3_start = prices.iloc[2]
        
        extensions = [1.618, 2.618, 4.236]
        ext_colors = ['#9370DB', '#8A2BE2', '#4B0082']
        
        for i, ext in enumerate(extensions):
            extension_level = wave_3_start + (wave_1_length * ext)
            ax.axhline(extension_level, color=ext_colors[i], linestyle='--', 
                      alpha=0.5, linewidth=1)
            
            if len(dates) > 3:
                text_date = dates[3]
                ax.text(text_date, extension_level, f'{ext:.1%}', 
                       fontsize=8, color=ext_colors[i], fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                       
    except Exception as e:
        logger.info(f"Warning: Could not add Fibonacci levels: {e}")


def add_info_box(ax: plt.Axes, info_text: str, position: str = 'top_left') -> None:
    """
    Add an information box to the chart.
    
    Args:
        ax: Matplotlib axes
        info_text: Text to display
        position: Position of the box ('top_left', 'top_right', 'bottom_left', 'bottom_right')
    """
    position_map = {
        'top_left': (0.02, 0.98),
        'top_right': (0.98, 0.98),
        'bottom_left': (0.02, 0.02),
        'bottom_right': (0.98, 0.02)
    }
    
    x, y = position_map.get(position, (0.02, 0.98))
    va = 'top' if 'top' in position else 'bottom'
    ha = 'left' if 'left' in position else 'right'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
    ax.text(x, y, info_text, transform=ax.transAxes, fontsize=11,
            verticalalignment=va, horizontalalignment=ha, bbox=props)


def add_relationship_annotations(ax: plt.Axes, relationships: Dict[str, Any], 
                               plotted_patterns: List[Dict[str, Any]]) -> None:
    """
    Add relationship annotations to the chart.
    
    Args:
        ax: Matplotlib axes
        relationships: Dictionary of pattern relationships
        plotted_patterns: List of plotted patterns
    """
    confirmations = relationships.get('confirmations', [])
    conflicts = relationships.get('conflicts', [])
    
    # Add confirmation arrows
    for confirmation in confirmations[:3]:  # Limit to first 3
        tf1 = confirmation['pattern1']['timeframe']
        tf2 = confirmation['pattern2']['timeframe']
        
        # Find corresponding plotted patterns
        pattern1 = next((p for p in plotted_patterns if p.get('timeframe') == tf1), None)
        pattern2 = next((p for p in plotted_patterns if p.get('timeframe') == tf2), None)
        
        if pattern1 and pattern2:
            # Draw arrow between patterns
            mid1 = len(pattern1['dates']) // 2
            mid2 = len(pattern2['dates']) // 2
            
            if mid1 < len(pattern1['dates']) and mid2 < len(pattern2['dates']):
                ax.annotate('', 
                           xy=(pattern2['dates'][mid2], pattern2['prices'].iloc[mid2]),
                           xytext=(pattern1['dates'][mid1], pattern1['prices'].iloc[mid1]),
                           arrowprops=dict(arrowstyle='->', color='green', lw=2, alpha=0.7))
    
    # Add conflict indicators
    for conflict in conflicts[:2]:  # Limit to first 2
        tf1 = conflict['pattern1']['timeframe']
        tf2 = conflict['pattern2']['timeframe']
        
        pattern1 = next((p for p in plotted_patterns if p.get('timeframe') == tf1), None)
        pattern2 = next((p for p in plotted_patterns if p.get('timeframe') == tf2), None)
        
        if pattern1 and pattern2:
            # Draw conflict indicator
            mid1 = len(pattern1['dates']) // 2
            mid2 = len(pattern2['dates']) // 2
            
            if mid1 < len(pattern1['dates']) and mid2 < len(pattern2['dates']):
                ax.annotate('⚠', 
                           xy=(pattern2['dates'][mid2], pattern2['prices'].iloc[mid2]),
                           xytext=(pattern1['dates'][mid1], pattern1['prices'].iloc[mid1]),
                           arrowprops=dict(arrowstyle='<->', color='red', lw=2, alpha=0.7),
                           fontsize=12, color='red', fontweight='bold')


def add_hierarchy_info_box(ax: plt.Axes, hierarchy: Dict[str, Any], 
                          relationships: Dict[str, Any]) -> None:
    """
    Add hierarchy information box to the chart.
    
    Args:
        ax: Matplotlib axes
        hierarchy: Pattern hierarchy dictionary
        relationships: Pattern relationships dictionary
    """
    info_text = "Pattern Hierarchy:\n"
    
    if hierarchy.get('primary'):
        primary = hierarchy['primary']
        info_text += f"Primary: {primary['timeframe']} ({primary['pattern']['confidence']:.1%})\n"
    
    supporting_count = len(hierarchy.get('supporting', []))
    if supporting_count > 0:
        info_text += f"Supporting: {supporting_count}\n"
    
    conflicting_count = len(hierarchy.get('conflicting', []))
    if conflicting_count > 0:
        info_text += f"Conflicting: {conflicting_count}\n"
    
    independent_count = len(hierarchy.get('independent', []))
    if independent_count > 0:
        info_text += f"Independent: {independent_count}\n"
    
    # Add relationship summary
    confirmations = len(relationships.get('confirmations', []))
    conflicts = len(relationships.get('conflicts', []))
    
    if confirmations > 0 or conflicts > 0:
        info_text += f"\nRelationships:\n"
        info_text += f"Confirmations: {confirmations}\n"
        info_text += f"Conflicts: {conflicts}\n"
    
    add_info_box(ax, info_text, 'top_right')


def add_trading_signals_box(ax: plt.Axes, signals: List[Dict[str, Any]]) -> None:
    """
    Add trading signals box to the chart.
    
    Args:
        ax: Matplotlib axes
        signals: List of trading signals
    """
    if not signals:
        return
    
    info_text = "Trading Signals:\n"
    
    for i, signal in enumerate(signals[:3]):  # Limit to first 3
        signal_type = signal.get('type', 'Unknown')
        confidence = signal.get('confidence', 0)
        direction = signal.get('direction', 'Unknown')
        
        info_text += f"{i+1}. {signal_type}: {direction} ({confidence:.1%})\n"
    
    if len(signals) > 3:
        info_text += f"... and {len(signals) - 3} more"
    
    add_info_box(ax, info_text, 'bottom_left')


def add_risk_assessment_box(ax: plt.Axes, risks: List[Dict[str, Any]]) -> None:
    """
    Add risk assessment box to the chart.
    
    Args:
        ax: Matplotlib axes
        risks: List of risk assessments
    """
    if not risks:
        return
    
    info_text = "Risk Assessment:\n"
    
    for i, risk in enumerate(risks[:3]):  # Limit to first 3
        risk_type = risk.get('type', 'Unknown')
        severity = risk.get('severity', 'Unknown')
        description = risk.get('description', 'No description')
        
        info_text += f"{i+1}. {risk_type}: {severity}\n"
        if len(description) > 30:
            description = description[:27] + "..."
        info_text += f"   {description}\n"
    
    if len(risks) > 3:
        info_text += f"... and {len(risks) - 3} more risks"
    
    add_info_box(ax, info_text, 'bottom_right')


def create_multi_panel_layout(fig: plt.Figure, n_panels: int = 4) -> List[plt.Axes]:
    """
    Create a multi-panel layout for comprehensive analysis.
    
    Args:
        fig: Matplotlib figure
        n_panels: Number of panels to create
    
    Returns:
        List of axes objects
    """
    if n_panels == 4:
        gs = fig.add_gridspec(2, 3, height_ratios=[2, 1], width_ratios=[2, 1, 1])
        
        # Main price chart (spans 2 rows, 1 column)
        ax_main = fig.add_subplot(gs[:, 0])
        
        # Pattern overview (top right)
        ax_overview = fig.add_subplot(gs[0, 1])
        
        # Volume analysis (top right, second row)
        ax_volume = fig.add_subplot(gs[0, 2])
        
        # Confidence meters (bottom right)
        ax_confidence = fig.add_subplot(gs[1, 1:])
        
        return [ax_main, ax_overview, ax_volume, ax_confidence]
    
    elif n_panels == 6:
        gs = fig.add_gridspec(3, 3, height_ratios=[2, 1, 1], width_ratios=[2, 1, 1])
        
        # Main price chart
        ax_main = fig.add_subplot(gs[:, 0])
        
        # Pattern overview
        ax_overview = fig.add_subplot(gs[0, 1])
        
        # Volume analysis
        ax_volume = fig.add_subplot(gs[0, 2])
        
        # Pattern timeline
        ax_timeline = fig.add_subplot(gs[1, 1:])
        
        # Confidence meters
        ax_confidence = fig.add_subplot(gs[2, 1:])
        
        return [ax_main, ax_overview, ax_volume, ax_timeline, ax_confidence]
    
    else:
        # Default single panel
        return [fig.add_subplot(111)]


def setup_professional_style() -> Dict[str, Any]:
    """
    Setup professional styling for charts.
    
    Returns:
        Style configuration dictionary
    """
    return {
        'figure_size': (16, 10),
        'dpi': 100,
        'font_size': {
            'title': 16,
            'label': 12,
            'tick': 10,
            'legend': 10
        },
        'colors': {
            'background': '#f8f9fa',
            'grid': '#e9ecef',
            'text': '#212529'
        },
        'line_widths': {
            'price': 1.0,
            'pattern': 2.0,
            'grid': 0.5
        },
        'alpha': {
            'price': 0.7,
            'pattern': 0.8,
            'grid': 0.3
        }
    }


def apply_professional_style(ax: plt.Axes, style_config: Dict[str, Any]) -> None:
    """
    Apply professional styling to an axes object.
    
    Args:
        ax: Matplotlib axes
        style_config: Style configuration dictionary
    """
    # Set background color
    ax.set_facecolor(style_config['colors']['background'])
    
    # Configure grid
    ax.grid(True, alpha=style_config['alpha']['grid'], 
            color=style_config['colors']['grid'], 
            linewidth=style_config['line_widths']['grid'])
    
    # Configure text colors
    ax.tick_params(colors=style_config['colors']['text'])
    ax.xaxis.label.set_color(style_config['colors']['text'])
    ax.yaxis.label.set_color(style_config['colors']['text'])
    ax.title.set_color(style_config['colors']['text'])
    
    # Configure font sizes
    ax.title.set_fontsize(style_config['font_size']['title'])
    ax.xaxis.label.set_fontsize(style_config['font_size']['label'])
    ax.yaxis.label.set_fontsize(style_config['font_size']['label'])
    ax.tick_params(labelsize=style_config['font_size']['tick'])


def create_confidence_meter(ax: plt.Axes, confidence: float, title: str = "Confidence") -> None:
    """
    Create a confidence meter visualization.
    
    Args:
        ax: Matplotlib axes
        confidence: Confidence value (0.0 to 1.0)
        title: Title for the meter
    """
    # Create horizontal bar
    colors = ['#DC143C', '#FFA500', '#2E8B57']  # Red, Orange, Green
    thresholds = [0.0, 0.4, 0.7, 1.0]
    
    # Determine color based on confidence
    color_idx = 0
    for i, threshold in enumerate(thresholds[1:], 1):
        if confidence >= threshold:
            color_idx = i - 1
    
    # Create the bar
    ax.barh(0, confidence, color=colors[color_idx], alpha=0.8, height=0.6)
    ax.barh(0, 1.0, color='lightgray', alpha=0.3, height=0.6)
    
    # Add confidence text
    ax.text(confidence + 0.02, 0, f'{confidence:.1%}', 
            va='center', ha='left', fontweight='bold', fontsize=10)
    
    # Configure axes
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.5, 0.5)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False) 