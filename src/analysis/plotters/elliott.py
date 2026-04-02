import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import matplotlib.cm as cm
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D
from typing import Dict, Any, Tuple, List
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Polygon
from matplotlib.collections import PatchCollection

from .visualization_utils import (
    fix_date_labels_enhanced,
    get_confidence_color,
    get_pattern_color_scheme,
    add_wave_labels,
    add_fibonacci_levels,
    add_info_box,
    add_relationship_annotations,
    add_hierarchy_info_box,
    add_trading_signals_box,
    add_risk_assessment_box,
    create_multi_panel_layout,
    setup_professional_style,
    apply_professional_style,
    create_confidence_meter
)

def plot_subwaves_recursive(ax, df, subwaves, depth=1, column='close', max_depth=3):
    if not subwaves or depth > max_depth:
        return
    colors = cm.get_cmap('tab10')
    for i, sub in enumerate(subwaves):
        if sub and 'sequence' in sub:
            points = [pt[0] for pt in sub['sequence']]
            if len(points) > 1:
                ax.plot(df.index[points], df[column].iloc[points],
                        marker='o', linestyle='-', linewidth=max(0.5, 2-depth*0.5),
                        color=colors((depth*2+i)%10), alpha=0.7,
                        label=f'Subwave d{depth}.{i+1}')
            # Recurse into deeper subwaves
            plot_subwaves_recursive(ax, df, sub.get('subwaves', []), depth+1, column, max_depth)

def plot_multiple_elliott_patterns_advanced(df: pd.DataFrame, 
                                          patterns_data: Dict[str, Any],
                                          column: str = 'close', 
                                          title: str = 'Advanced Multi-Pattern Elliott Wave Analysis',
                                          ax=None, 
                                          show_relationships: bool = True,
                                          show_hierarchy: bool = True) -> matplotlib.figure.Figure:
    """
    Advanced plotting function for multiple Elliott Wave patterns with relationships
    
    Args:
        df: DataFrame with price data
        patterns_data: Dictionary containing multiple patterns and relationships
        column: Column name for price data
        title: Chart title
        ax: Matplotlib axes (if None, creates new figure)
        show_relationships: Whether to show pattern relationships
        show_hierarchy: Whether to show pattern hierarchy
    
    Returns:
        Matplotlib figure object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(18, 12))
    else:
        fig = ax.figure
    
    # Extract data
    composite_patterns = patterns_data.get('composite_patterns', [])
    pattern_hierarchy = patterns_data.get('pattern_hierarchy', {})
    pattern_relationships = patterns_data.get('pattern_relationships', {})
    
    # Color scheme for different pattern types
    color_scheme = {
        'primary': '#2E86AB',      # Blue for primary patterns
        'supporting': '#A23B72',   # Purple for supporting patterns
        'conflicting': '#F18F01',  # Orange for conflicting patterns
        'independent': '#C73E1D',  # Red for independent patterns
        'alternative': '#8B4513'   # Brown for alternative patterns
    }
    
    # Plot price line
    ax.plot(df.index, df[column], 'k-', linewidth=1, alpha=0.6, label='Price', zorder=1)
    
    # Plot patterns by hierarchy
    plotted_patterns = []
    
    # Plot primary pattern first (most prominent)
    if pattern_hierarchy.get('primary'):
        primary = pattern_hierarchy['primary']
        pattern = primary['pattern']
        points = pattern.get('points', [])
        
        if len(points) >= 2:
            dates = df.index[points]
            prices = df[column].iloc[points]
            
            ax.plot(dates, prices, 
                   color=color_scheme['primary'], linewidth=4, alpha=0.9,
                   marker='o', markersize=12, markerfacecolor='white',
                   markeredgecolor=color_scheme['primary'], markeredgewidth=3,
                   label=f"Primary Pattern ({primary['timeframe']}, {pattern['confidence']:.1%})",
                   zorder=5)
            
            # Add wave labels for primary pattern
            wave_labels = ['S', '1', '2', '3', '4', '5'][:len(points)]
            for i, (date, price, label) in enumerate(zip(dates, prices, wave_labels)):
                ax.annotate(label, (date, price), 
                           xytext=(8, 12), textcoords='offset points',
                           fontsize=12, fontweight='bold', 
                           color=color_scheme['primary'],
                           bbox=dict(boxstyle="round,pad=0.3", 
                                   facecolor='white', 
                                   edgecolor=color_scheme['primary'],
                                   alpha=0.8))
            
            plotted_patterns.append({
                'pattern': pattern,
                'type': 'primary',
                'timeframe': primary['timeframe'],
                'dates': dates,
                'prices': prices
            })
    
    # Plot supporting patterns
    for i, supporting in enumerate(pattern_hierarchy.get('supporting', [])):
        pattern = supporting['pattern']
        points = pattern.get('points', [])
        
        if len(points) >= 2:
            dates = df.index[points]
            prices = df[column].iloc[points]
            
            ax.plot(dates, prices, 
                   color=color_scheme['supporting'], linewidth=3, alpha=0.7,
                   marker='s', markersize=10, markerfacecolor='white',
                   markeredgecolor=color_scheme['supporting'], markeredgewidth=2,
                   label=f"Supporting {i+1} ({supporting['timeframe']}, {pattern['confidence']:.1%})",
                   zorder=4)
            
            plotted_patterns.append({
                'pattern': pattern,
                'type': 'supporting',
                'timeframe': supporting['timeframe'],
                'dates': dates,
                'prices': prices
            })
    
    # Plot conflicting patterns
    for i, conflicting in enumerate(pattern_hierarchy.get('conflicting', [])):
        pattern = conflicting['pattern']
        points = pattern.get('points', [])
        
        if len(points) >= 2:
            dates = df.index[points]
            prices = df[column].iloc[points]
            
            ax.plot(dates, prices, 
                   color=color_scheme['conflicting'], linewidth=2, alpha=0.6,
                   marker='^', markersize=8, markerfacecolor='white',
                   markeredgecolor=color_scheme['conflicting'], markeredgewidth=2,
                   label=f"Conflicting {i+1} ({conflicting['timeframe']}, {pattern['confidence']:.1%})",
                   zorder=3)
            
            plotted_patterns.append({
                'pattern': pattern,
                'type': 'conflicting',
                'timeframe': conflicting['timeframe'],
                'dates': dates,
                'prices': prices
            })
    
    # Plot independent patterns
    for i, independent in enumerate(pattern_hierarchy.get('independent', [])):
        pattern = independent['pattern']
        points = pattern.get('points', [])
        
        if len(points) >= 2:
            dates = df.index[points]
            prices = df[column].iloc[points]
            
            ax.plot(dates, prices, 
                   color=color_scheme['independent'], linewidth=2, alpha=0.5,
                   marker='d', markersize=6, markerfacecolor='white',
                   markeredgecolor=color_scheme['independent'], markeredgewidth=1,
                   label=f"Independent {i+1} ({independent['timeframe']}, {pattern['confidence']:.1%})",
                   zorder=2)
            
            plotted_patterns.append({
                'pattern': pattern,
                'type': 'independent',
                'timeframe': independent['timeframe'],
                'dates': dates,
                'prices': prices
            })
    
    # Add relationship annotations if requested
    if show_relationships and pattern_relationships:
        add_relationship_annotations(ax, pattern_relationships, plotted_patterns)
    
    # Add hierarchy information box
    if show_hierarchy:
        add_hierarchy_info_box(ax, pattern_hierarchy, pattern_relationships)
    
    # Add trading signals if available
    trading_signals = pattern_relationships.get('trading_signals', [])
    if trading_signals:
        add_trading_signals_box(ax, trading_signals)
    
    # Add risk assessments if available
    risk_assessments = pattern_relationships.get('risk_assessments', [])
    if risk_assessments:
        add_risk_assessment_box(ax, risk_assessments)
    
    # Formatting
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10, 
             framealpha=0.8, borderaxespad=0)
    
    # Fix date labels
    fix_date_labels_enhanced(ax, df)
    
    fig.tight_layout()
    return fig

def plot_pattern_comparison_chart(df: pd.DataFrame, 
                                patterns_data: Dict[str, Any],
                                column: str = 'close') -> matplotlib.figure.Figure:
    """
    Create a comparison chart showing patterns across different timeframes
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Elliott Wave Pattern Comparison Across Timeframes', fontsize=16, fontweight='bold')
    
    patterns_by_timeframe = patterns_data.get('patterns_by_timeframe', {})
    timeframes = list(patterns_by_timeframe.keys())
    
    # Plot each timeframe in a subplot
    for i, timeframe in enumerate(timeframes[:4]):  # Limit to 4 timeframes
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        timeframe_data = patterns_by_timeframe[timeframe]
        multiple_patterns = timeframe_data.get('multiple_patterns', [])
        
        # Plot price
        ax.plot(df.index, df[column], 'k-', linewidth=1, alpha=0.6)
        
        # Plot patterns for this timeframe
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        for j, pattern in enumerate(multiple_patterns[:3]):  # Limit to 3 patterns per timeframe
            points = pattern.get('points', [])
            if len(points) >= 2:
                dates = df.index[points]
                prices = df[column].iloc[points]
                
                ax.plot(dates, prices, 
                       color=colors[j % len(colors)], linewidth=2, alpha=0.8,
                       marker='o', markersize=6,
                       label=f"Pattern {j+1} ({pattern['confidence']:.1%})")
        
        ax.set_title(f'{timeframe.title()} Timeframe', fontsize=12, fontweight='bold')
        ax.set_ylabel('Price ($)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Hide unused subplots
    for i in range(len(timeframes), 4):
        row = i // 2
        col = i % 2
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    return fig

def plot_elliott_wave_analysis_enhanced(df: pd.DataFrame, wave_data: Dict[str, Any],
                                      column: str = 'close', title: str = 'Elliott Wave Analysis',
                                      ax=None, show_validation_details: bool = True):
    """
    Enhanced plotting with comprehensive validation details and improved visualization.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 10))
    else:
        fig = ax.figure
    
    # Extract wave data
    impulse_wave = wave_data.get('impulse_wave', np.array([]))
    corrective_wave = wave_data.get('corrective_wave', np.array([]))
    confidence = wave_data.get('confidence', 0.0)
    wave_type = wave_data.get('wave_type', 'unknown')
    validation_details = wave_data.get('validation_details', {})
    
    # Color scheme
    impulse_color = '#2E86AB'  # Blue for impulse waves
    corrective_color = '#E63946'  # Red for corrective waves
    wave_line_color = '#F77F00'  # Orange for wave connections
    
    # Plot price line thinly
    ax.plot(df.index, df[column], 'k-', linewidth=0.5, alpha=0.5, label='Price')
    
    # Plot impulse wave structure
    if len(impulse_wave) >= 2:
        impulse_prices = df[column].iloc[impulse_wave].values
        impulse_dates = df.index[impulse_wave]
        
        # Draw wave connections with thicker lines
        ax.plot(impulse_dates, impulse_prices, 
                color=wave_line_color, linewidth=3, alpha=0.8, 
                marker='o', markersize=10, markerfacecolor='white',
                markeredgecolor=impulse_color, markeredgewidth=3,
                label='Impulse Wave (1-2-3-4-5)', zorder=5)
        
        # Label each wave point
        wave_labels = ['Start', '1', '2', '3', '4', '5']
        for i, (date, price, label) in enumerate(zip(impulse_dates, impulse_prices, wave_labels)):
            # Determine if point is peak or trough for better label positioning
            if i == 0:
                offset = 0
            elif i % 2 == 1:  # Odd waves (1, 3, 5) - typically peaks in uptrend
                offset = price * 0.02 if impulse_prices[1] > impulse_prices[0] else -price * 0.02
            else:  # Even waves (2, 4) - typically troughs in uptrend
                offset = -price * 0.02 if impulse_prices[1] > impulse_prices[0] else price * 0.02
            
            # Wave number in circle
            circle = plt.Circle((mdates.date2num(date), price), 
                              radius=0.015*(ax.get_ylim()[1]-ax.get_ylim()[0]), 
                              color='white', zorder=6)
            ax.add_patch(circle)
            ax.text(date, price, label if label != 'Start' else 'S', 
                    fontsize=10, fontweight='bold', color=impulse_color,
                    ha='center', va='center', zorder=7)
            
            # Additional label with price
            ax.text(date, price + offset, f'${price:.2f}', 
                    fontsize=8, ha='center', va='bottom' if offset > 0 else 'top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                             edgecolor=impulse_color, alpha=0.8))
        
        # Draw wave structure guidelines
        if len(impulse_wave) >= 5:
            # Trend channel lines
            # Upper trend line (connecting waves 1-3-5)
            upper_points = [0, 2, 4] if len(impulse_wave) > 4 else [0, 2]
            upper_dates = [impulse_dates[i] for i in upper_points if i < len(impulse_dates)]
            upper_prices = [impulse_prices[i] for i in upper_points if i < len(impulse_prices)]
            
            if len(upper_dates) >= 2:
                z = np.polyfit([mdates.date2num(d) for d in upper_dates], upper_prices, 1)
                p = np.poly1d(z)
                extended_dates = pd.date_range(start=impulse_dates[0], 
                                             end=impulse_dates[-1], freq='D')
                ax.plot(extended_dates, p([mdates.date2num(d) for d in extended_dates]), 
                        '--', color=impulse_color, alpha=0.5, linewidth=1, 
                        label='Impulse Channel')
            
            # Lower trend line (connecting waves 2-4)
            if len(impulse_wave) >= 5:
                lower_dates = [impulse_dates[2], impulse_dates[4]]
                lower_prices = [impulse_prices[2], impulse_prices[4]]
                z = np.polyfit([mdates.date2num(d) for d in lower_dates], lower_prices, 1)
                p = np.poly1d(z)
                ax.plot(extended_dates, p([mdates.date2num(d) for d in extended_dates]), 
                        '--', color=impulse_color, alpha=0.5, linewidth=1)
    
    # Plot corrective wave structure
    if len(corrective_wave) >= 2:
        corrective_prices = df[column].iloc[corrective_wave].values
        corrective_dates = df.index[corrective_wave]
        
        # Draw corrective wave connections
        ax.plot(corrective_dates, corrective_prices, 
                color=corrective_color, linewidth=3, alpha=0.8,
                marker='s', markersize=10, markerfacecolor='white',
                markeredgecolor=corrective_color, markeredgewidth=3,
                label='Corrective Wave (A-B-C)', zorder=5)
        
        # Label corrective waves
        wave_labels = ['A', 'B', 'C', 'End'][:len(corrective_wave)]
        for date, price, label in zip(corrective_dates, corrective_prices, wave_labels):
            # Wave letter in square
            square = FancyBboxPatch((mdates.date2num(date)-0.5, price-price*0.01), 
                                  1, price*0.02, boxstyle="round,pad=0.1",
                                  facecolor='white', edgecolor=corrective_color, 
                                  linewidth=2, zorder=6)
            ax.add_patch(square)
            ax.text(date, price, label, fontsize=10, fontweight='bold', 
                    color=corrective_color, ha='center', va='center', zorder=7)
    
    # Add wave analysis information box
    info_text = f"Wave Pattern: {wave_type}\n"
    info_text += f"Confidence: {confidence:.1%}\n"
    info_text += f"Impulse Waves: {len(impulse_wave)} points\n"
    info_text += f"Corrective Waves: {len(corrective_wave)} points"
    
    # Add pattern description
    if len(impulse_wave) >= 5:
        trend = "Bullish" if df[column].iloc[impulse_wave[-1]] > df[column].iloc[impulse_wave[0]] else "Bearish"
        info_text += f"\nTrend: {trend}"
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    # Add Fibonacci retracement levels if we have complete impulse wave
    if len(impulse_wave) >= 5:
        add_fibonacci_levels(ax, df, impulse_wave, column)
    
    # Enhanced title and labels
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10, framealpha=0.6, borderaxespad=0)
    
    # Fix date labels
    fix_date_labels_enhanced(ax, df)
    
    fig.tight_layout()
    if ax is None:
        plt.show()

def plot_elliott_wave_analysis(df: pd.DataFrame, wave_data: Dict[str, Any], column: str = 'close',
                              title: str = 'Elliott Wave Analysis', ax=None):
    """
    Basic Elliott Wave analysis plotting function.
    """
    return plot_elliott_wave_analysis_enhanced(df, wave_data, column, title, ax)

class MultiPatternVisualizer:
    """Enhanced visualizer for multiple Elliott Wave patterns"""
    
    def __init__(self, style_config: Dict[str, Any] = None):
        self.style_config = style_config or self._get_default_style()
        self.color_palette = self._setup_color_palette()
        
    def _get_default_style(self) -> Dict[str, Any]:
        """Get default style configuration"""
        return {
            'figure_size': (16, 10),
            'dpi': 100,
            'background_color': 'white',
            'grid_alpha': 0.3,
            'primary_line_width': 3,
            'secondary_line_width': 2,
            'marker_size': 10,
            'font_size': {
                'title': 16,
                'label': 12,
                'annotation': 10,
                'legend': 10
            },
            'panel_ratios': (4, 1, 1)  # Price, Volume, Indicators
        }
    
    def _setup_color_palette(self) -> Dict[str, str]:
        """Setup color palette for different pattern types"""
        return {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'tertiary': '#F18F01',
            'success': '#27AE60',
            'warning': '#F39C12',
            'danger': '#E74C3C',
            'neutral': '#95A5A6',
            'impulse': '#3498DB',
            'corrective': '#E67E22',
            'diagonal': '#9B59B6',
            'background': '#ECF0F1'
        }
    
    def plot_multi_pattern_analysis(self,
                                  df: pd.DataFrame,
                                  patterns_data: Dict[str, Any],
                                  title: str = 'Multi-Pattern Elliott Wave Analysis',
                                  show_interactions: bool = True,
                                  show_projections: bool = True) -> plt.Figure:
        """
        Create comprehensive multi-pattern visualization
        
        Args:
            df: Price data
            patterns_data: Dictionary containing patterns and relationships
            title: Plot title
            show_interactions: Whether to show pattern interactions
            show_projections: Whether to show future projections
        
        Returns:
            Matplotlib figure
        """
        # Create figure with subplots
        fig = plt.figure(figsize=self.style_config['figure_size'], 
                        dpi=self.style_config['dpi'])
        
        # Create grid layout
        gs = fig.add_gridspec(4, 4, height_ratios=[3, 1, 1, 1], 
                             width_ratios=[3, 1, 1, 1],
                             hspace=0.3, wspace=0.3)
        
        # Main price chart
        ax_main = fig.add_subplot(gs[0, :3])
        
        # Pattern overview panel
        ax_overview = fig.add_subplot(gs[0, 3])
        
        # Volume
        ax_volume = fig.add_subplot(gs[1, :3], sharex=ax_main)
        
        # Pattern timeline
        ax_timeline = fig.add_subplot(gs[2, :3], sharex=ax_main)
        
        # Confidence meters
        ax_confidence = fig.add_subplot(gs[3, :3])
        
        # Relationship diagram
        ax_relationships = fig.add_subplot(gs[1:, 3])
        
        # Plot components
        self._plot_price_with_patterns(ax_main, df, patterns_data)
        self._plot_pattern_overview(ax_overview, patterns_data)
        self._plot_volume_analysis(ax_volume, df, patterns_data)
        self._plot_pattern_timeline(ax_timeline, patterns_data)
        self._plot_confidence_meters(ax_confidence, patterns_data)
        
        if show_interactions:
            self._plot_relationship_diagram(ax_relationships, patterns_data)
        
        if show_projections:
            self._add_projections(ax_main, df, patterns_data)
        
        # Overall styling
        fig.suptitle(title, fontsize=self.style_config['font_size']['title'], 
                    fontweight='bold')
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        return fig
    
    def _plot_price_with_patterns(self,
                                ax: plt.Axes,
                                df: pd.DataFrame,
                                patterns_data: Dict[str, Any]) -> None:
        """Plot price data with multiple patterns overlaid"""
        # Plot base price line
        ax.plot(df.index, df['close'], 'k-', linewidth=0.5, alpha=0.5, label='Price')
        
        # Extract patterns
        patterns = patterns_data.get('composite_patterns', [])
        
        # Plot patterns with different styles
        for i, pattern_info in enumerate(patterns[:5]):  # Limit to 5 patterns
            pattern = pattern_info.get('base_pattern', {}).get('pattern', {})
            wave_points = pattern.get('points', [])
            
            if len(wave_points) < 2:
                continue
            
            # Determine style based on pattern rank
            if i == 0:  # Primary pattern
                color = self.color_palette['primary']
                linewidth = self.style_config['primary_line_width']
                alpha = 1.0
                linestyle = '-'
                zorder = 10
            else:  # Secondary patterns
                colors = [self.color_palette['secondary'], 
                         self.color_palette['tertiary'],
                         self.color_palette['warning'],
                         self.color_palette['neutral']]
                color = colors[min(i-1, len(colors)-1)]
                linewidth = self.style_config['secondary_line_width']
                alpha = 0.7
                linestyle = ['--', '-.', ':', '-'][min(i-1, 3)]
                zorder = 9 - i
            
            # Plot pattern line
            pattern_dates = df.index[wave_points]
            pattern_prices = df['close'].iloc[wave_points]
            
            ax.plot(pattern_dates, pattern_prices,
                   color=color, linewidth=linewidth, alpha=alpha,
                   linestyle=linestyle, marker='o', markersize=8,
                   label=f"Pattern {i+1} ({pattern_info.get('composite_confidence', 0):.1%})",
                   zorder=zorder)
            
            # Add wave labels for primary pattern
            if i == 0:
                self._add_wave_labels(ax, pattern_dates, pattern_prices, wave_points)
        
        # Add pattern conflict zones
        self._highlight_conflict_zones(ax, df, patterns_data)
        
        # Styling
        ax.set_ylabel('Price', fontsize=self.style_config['font_size']['label'])
        ax.grid(True, alpha=self.style_config['grid_alpha'])
        ax.legend(loc='upper left', fontsize=self.style_config['font_size']['legend'])
        
        # Format dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    
    def _add_wave_labels(self,
                        ax: plt.Axes,
                        dates: pd.DatetimeIndex,
                        prices: pd.Series,
                        wave_points: List[int]) -> None:
        """Add wave number labels to pattern"""
        for i, (date, price) in enumerate(zip(dates, prices)):
            # Determine if peak or trough for positioning
            is_peak = i % 2 == 0
            offset = 20 if is_peak else -20
            
            # Create fancy label
            bbox_props = dict(boxstyle="round,pad=0.3", 
                            facecolor='white', 
                            edgecolor=self.color_palette['primary'],
                            linewidth=2)
            
            ax.annotate(f'{i+1}',
                       xy=(date, price),
                       xytext=(0, offset),
                       textcoords='offset points',
                       ha='center',
                       va='bottom' if is_peak else 'top',
                       fontsize=self.style_config['font_size']['annotation'],
                       fontweight='bold',
                       color=self.color_palette['primary'],
                       bbox=bbox_props,
                       arrowprops=dict(arrowstyle='-',
                                     color=self.color_palette['primary'],
                                     lw=1))
    
    def _highlight_conflict_zones(self,
                                ax: plt.Axes,
                                df: pd.DataFrame,
                                patterns_data: Dict[str, Any]) -> None:
        """Highlight areas where patterns conflict"""
        relationships = patterns_data.get('pattern_relationships', {})
        conflicts = relationships.get('conflicts', [])
        
        for conflict in conflicts:
            # Get conflict time range
            # This is simplified - in practice would calculate actual overlap
            start_idx = len(df) // 2
            end_idx = min(start_idx + 50, len(df) - 1)
            
            # Add shaded region
            ax.axvspan(df.index[start_idx], df.index[end_idx],
                      alpha=0.1, color=self.color_palette['danger'],
                      label='Conflict Zone' if conflicts.index(conflict) == 0 else '')
    
    def _plot_pattern_overview(self,
                             ax: plt.Axes,
                             patterns_data: Dict[str, Any]) -> None:
        """Plot pattern overview panel"""
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.95, 'Pattern Overview',
               transform=ax.transAxes,
               fontsize=self.style_config['font_size']['label'],
               fontweight='bold',
               ha='center')
        
        # Pattern summary
        patterns = patterns_data.get('composite_patterns', [])
        y_position = 0.85
        
        for i, pattern_info in enumerate(patterns[:5]):
            confidence = pattern_info.get('composite_confidence', 0)
            support_count = pattern_info.get('support_count', 0)
            
            # Pattern indicator
            color = self._get_confidence_color(confidence)
            ax.add_patch(Circle((0.1, y_position), 0.03,
                               color=color, transform=ax.transAxes))
            
            # Pattern text
            text = f"P{i+1}: {confidence:.0%}"
            if support_count > 0:
                text += f" (+{support_count})"
            
            ax.text(0.2, y_position, text,
                   transform=ax.transAxes,
                   fontsize=self.style_config['font_size']['annotation'],
                   va='center')
            
            y_position -= 0.15
    
    def _plot_volume_analysis(self,
                            ax: plt.Axes,
                            df: pd.DataFrame,
                            patterns_data: Dict[str, Any]) -> None:
        """Plot volume with pattern-specific analysis"""
        if 'volume' not in df.columns:
            ax.text(0.5, 0.5, 'No volume data available',
                   transform=ax.transAxes, ha='center', va='center')
            return
        
        # Plot volume bars
        ax.bar(df.index, df['volume'], alpha=0.3, color='gray')
        
        # Highlight volume at pattern points
        patterns = patterns_data.get('composite_patterns', [])
        
        for i, pattern_info in enumerate(patterns[:2]):  # Top 2 patterns
            pattern = pattern_info.get('base_pattern', {}).get('pattern', {})
            wave_points = pattern.get('points', [])
            
            if len(wave_points) > 0:
                colors = [self.color_palette['primary'], self.color_palette['secondary']]
                ax.bar(df.index[wave_points], df['volume'].iloc[wave_points],
                      alpha=0.7, color=colors[i])
        
        ax.set_ylabel('Volume', fontsize=self.style_config['font_size']['label'])
        ax.grid(True, alpha=self.style_config['grid_alpha'])
    
    def _plot_pattern_timeline(self,
                             ax: plt.Axes,
                             patterns_data: Dict[str, Any]) -> None:
        """Plot timeline showing when different patterns are active"""
        patterns = patterns_data.get('patterns_by_timeframe', {})
        
        y_labels = []
        y_positions = []
        
        for i, (timeframe, tf_data) in enumerate(patterns.items()):
            y_labels.append(timeframe.capitalize())
            y_positions.append(i)
            
            for pattern in tf_data.get('multiple_patterns', [])[:3]:
                start_date = pattern.get('start_date')
                end_date = pattern.get('end_date')
                confidence = pattern.get('confidence', 0)
                
                if start_date and end_date:
                    # Create timeline bar
                    bar = Rectangle((mdates.date2num(start_date), i - 0.4),
                                  mdates.date2num(end_date) - mdates.date2num(start_date),
                                  0.8,
                                  facecolor=self._get_confidence_color(confidence),
                                  alpha=0.7,
                                  edgecolor='black',
                                  linewidth=1)
                    ax.add_patch(bar)
        
        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels)
        ax.set_ylim(-0.5, len(y_labels) - 0.5)
        ax.set_xlabel('Date', fontsize=self.style_config['font_size']['label'])
        ax.grid(True, alpha=self.style_config['grid_alpha'], axis='x')
        ax.set_title('Pattern Timeline', fontsize=self.style_config['font_size']['label'])
    
    def _plot_confidence_meters(self,
                              ax: plt.Axes,
                              patterns_data: Dict[str, Any]) -> None:
        """Plot confidence meters for different aspects"""
        metrics = {
            'Overall': patterns_data.get('pattern_relationships', {}).get('alignment_score', 0.5),
            'Primary': patterns_data.get('confidence', 0),
            'Convergence': len(patterns_data.get('pattern_relationships', {}).get('convergences', [])) / 10,
            'Timeframe': len(patterns_data.get('patterns_by_timeframe', {})) / 5
        }
        
        x_positions = np.arange(len(metrics))
        bar_width = 0.6
        
        bars = ax.bar(x_positions, list(metrics.values()), bar_width)
        
        # Color bars based on value
        for i, (bar, value) in enumerate(zip(bars, metrics.values())):
            bar.set_facecolor(self._get_confidence_color(value))
            
            # Add value label
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{value:.0%}',
                   ha='center', va='bottom',
                   fontsize=self.style_config['font_size']['annotation'])
        
        ax.set_xticks(x_positions)
        ax.set_xticklabels(list(metrics.keys()))
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Score', fontsize=self.style_config['font_size']['label'])
        ax.set_title('Confidence Metrics', fontsize=self.style_config['font_size']['label'])
        ax.grid(True, alpha=self.style_config['grid_alpha'], axis='y')
    
    def _plot_relationship_diagram(self,
                                 ax: plt.Axes,
                                 patterns_data: Dict[str, Any]) -> None:
        """Plot pattern relationship diagram"""
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.95, 'Pattern Relationships',
               transform=ax.transAxes,
               fontsize=self.style_config['font_size']['label'],
               fontweight='bold',
               ha='center')
        
        relationships = patterns_data.get('pattern_relationships', {})
        
        # Count relationship types
        confirmations = len(relationships.get('confirmations', []))
        conflicts = len(relationships.get('conflicts', []))
        nested = len(relationships.get('nested_patterns', []))
        
        # Create simple visualization
        y_pos = 0.8
        spacing = 0.2
        
        # Confirmations
        ax.text(0.1, y_pos, '✓ Confirmations:',
               transform=ax.transAxes,
               fontsize=self.style_config['font_size']['annotation'],
               color=self.color_palette['success'])
        ax.text(0.7, y_pos, str(confirmations),
               transform=ax.transAxes,
               fontsize=self.style_config['font_size']['annotation'],
               fontweight='bold')
        
        # Conflicts
        y_pos -= spacing
        ax.text(0.1, y_pos, '✗ Conflicts:',
               transform=ax.transAxes,
               fontsize=self.style_config['font_size']['annotation'],
               color=self.color_palette['danger'])
        ax.text(0.7, y_pos, str(conflicts),
               transform=ax.transAxes,
               fontsize=self.style_config['font_size']['annotation'],
               fontweight='bold')
        
        # Nested
        y_pos -= spacing
        ax.text(0.1, y_pos, '◐ Nested:',
               transform=ax.transAxes,
               fontsize=self.style_config['font_size']['annotation'],
               color=self.color_palette['warning'])
        ax.text(0.7, y_pos, str(nested),
               transform=ax.transAxes,
               fontsize=self.style_config['font_size']['annotation'],
               fontweight='bold')
        
        # Add visual diagram if space permits
        if confirmations + conflicts + nested > 0:
            self._add_relationship_visual(ax, relationships)
    
    def _add_relationship_visual(self,
                               ax: plt.Axes,
                               relationships: Dict[str, Any]) -> None:
        """Add visual representation of relationships"""
        # Create a simple network diagram
        center_x, center_y = 0.5, 0.3
        radius = 0.15
        
        # Primary pattern at center
        primary_circle = Circle((center_x, center_y), 0.05,
                              color=self.color_palette['primary'],
                              transform=ax.transAxes,
                              zorder=10)
        ax.add_patch(primary_circle)
        
        # Related patterns around it
        confirmations = relationships.get('confirmations', [])[:4]
        angle_step = 2 * np.pi / max(len(confirmations), 1)
        
        for i, confirmation in enumerate(confirmations):
            angle = i * angle_step
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            
            # Pattern circle
            circle = Circle((x, y), 0.03,
                          color=self.color_palette['success'],
                          transform=ax.transAxes,
                          alpha=0.7)
            ax.add_patch(circle)
            
            # Connection line
            ax.plot([center_x, x], [center_y, y],
                   color=self.color_palette['success'],
                   transform=ax.transAxes,
                   alpha=0.5,
                   linewidth=2)
    
    def _add_projections(self,
                       ax: plt.Axes,
                       df: pd.DataFrame,
                       patterns_data: Dict[str, Any]) -> None:
        """Add future projections based on pattern analysis"""
        patterns = patterns_data.get('composite_patterns', [])
        
        if not patterns:
            return
        
        # Get primary pattern for projection
        primary_pattern = patterns[0]
        pattern = primary_pattern.get('base_pattern', {}).get('pattern', {})
        wave_points = pattern.get('points', [])
        
        if len(wave_points) < 5:
            return
        
        # Calculate projection based on wave structure
        prices = df['close'].iloc[wave_points].values
        dates = df.index[wave_points]
        
        # Simple projection: extend the trend
        if len(prices) >= 5:
            # Calculate trend from wave 1 to wave 5
            trend_slope = (prices[4] - prices[0]) / (wave_points[4] - wave_points[0])
            
            # Project forward by 20% of the pattern duration
            projection_length = int((wave_points[4] - wave_points[0]) * 0.2)
            projection_end_idx = min(wave_points[4] + projection_length, len(df) - 1)
            
            if projection_end_idx > wave_points[4]:
                projection_price = prices[4] + trend_slope * (projection_end_idx - wave_points[4])
                
                # Draw projection line
                ax.plot([dates[4], df.index[projection_end_idx]], 
                       [prices[4], projection_price],
                       '--', color=self.color_palette['warning'],
                       linewidth=2, alpha=0.8,
                       label='Projection')
                
                # Add projection point
                ax.plot(df.index[projection_end_idx], projection_price,
                       'o', color=self.color_palette['warning'],
                       markersize=10, markeredgecolor='white',
                       markeredgewidth=2,
                       label='Projected Target')

    def create_pattern_comparison_plot(self,
                                     df: pd.DataFrame,
                                     patterns_list: List[Dict[str, Any]],
                                     title: str = 'Pattern Comparison') -> plt.Figure:
        """Create side-by-side comparison of multiple patterns"""
        n_patterns = min(len(patterns_list), 4)
        
        fig, axes = plt.subplots(n_patterns, 1, 
                                figsize=(self.style_config['figure_size'][0], 
                                        n_patterns * 3),
                                sharex=True)
        
        if n_patterns == 1:
            axes = [axes]
        
        for i, (ax, pattern_data) in enumerate(zip(axes, patterns_list[:n_patterns])):
            # Plot price
            ax.plot(df.index, df['close'], 'k-', alpha=0.3, linewidth=1)
            
            # Plot pattern
            wave_points = pattern_data.get('points', [])
            if len(wave_points) >= 2:
                ax.plot(df.index[wave_points], df['close'].iloc[wave_points],
                       'o-', linewidth=3, markersize=8,
                       label=f"Pattern {i+1}")
            
            # Add pattern info
            info_text = (f"Timeframe: {pattern_data.get('time_frame', 'Unknown')}\n"
                        f"Confidence: {pattern_data.get('confidence', 0):.1%}\n"
                        f"Type: {pattern_data.get('wave_type', 'Unknown')}")
            
            ax.text(0.02, 0.98, info_text,
                   transform=ax.transAxes,
                   fontsize=self.style_config['font_size']['annotation'],
                   verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.3',
                           facecolor='white',
                           alpha=0.8))
            
            ax.set_ylabel(f'Pattern {i+1}', 
                         fontsize=self.style_config['font_size']['label'])
            ax.grid(True, alpha=self.style_config['grid_alpha'])
            ax.legend()
        
        axes[-1].set_xlabel('Date', fontsize=self.style_config['font_size']['label'])
        fig.suptitle(title, fontsize=self.style_config['font_size']['title'], fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def create_interactive_dashboard(self,
                                   df: pd.DataFrame,
                                   patterns_data: Dict[str, Any]) -> Dict[str, plt.Figure]:
        """Create multiple figures for an interactive dashboard"""
        dashboard = {}
        
        # Main analysis
        dashboard['main'] = self.plot_multi_pattern_analysis(
            df, patterns_data, 
            title='Complete Elliott Wave Analysis'
        )
        
        # Pattern comparison
        patterns_list = patterns_data.get('multiple_patterns', [])
        if patterns_list:
            dashboard['comparison'] = self.create_pattern_comparison_plot(
                df, patterns_list,
                title='Individual Pattern Analysis'
            )
        
        # Timeframe analysis
        dashboard['timeframe'] = self.create_timeframe_analysis_plot(
            df, patterns_data
        )
        
        # Confidence analysis
        dashboard['confidence'] = self.create_confidence_analysis_plot(
            patterns_data
        )
        
        return dashboard
    
    def create_timeframe_analysis_plot(self,
                                     df: pd.DataFrame,
                                     patterns_data: Dict[str, Any]) -> plt.Figure:
        """Create detailed timeframe analysis plot"""
        patterns_by_tf = patterns_data.get('patterns_by_timeframe', {})
        n_timeframes = len(patterns_by_tf)
        
        if n_timeframes == 0:
            fig, ax = plt.subplots(figsize=self.style_config['figure_size'])
            ax.text(0.5, 0.5, 'No timeframe data available',
                   transform=ax.transAxes, ha='center', va='center')
            return fig
        
        fig, axes = plt.subplots(n_timeframes, 1,
                               figsize=(self.style_config['figure_size'][0],
                                       n_timeframes * 3),
                               sharex=True)
        
        if n_timeframes == 1:
            axes = [axes]
        
        colors = list(self.color_palette.values())
        
        for i, (timeframe, tf_data) in enumerate(patterns_by_tf.items()):
            ax = axes[i]
            
            # Plot price
            ax.plot(df.index, df['close'], 'k-', alpha=0.3, linewidth=1)
            
            # Plot patterns for this timeframe
            for j, pattern in enumerate(tf_data.get('multiple_patterns', [])[:3]):
                wave_points = pattern.get('points', [])
                if len(wave_points) >= 2:
                    color = colors[j % len(colors)]
                    ax.plot(df.index[wave_points], df['close'].iloc[wave_points],
                           'o-', color=color, linewidth=2, markersize=6,
                           alpha=0.8 - j*0.2,
                           label=f"{pattern.get('confidence', 0):.1%}")
            
            ax.set_title(f'{timeframe.capitalize()} Timeframe',
                        fontsize=self.style_config['font_size']['label'])
            ax.set_ylabel('Price', fontsize=self.style_config['font_size']['label'])
            ax.grid(True, alpha=self.style_config['grid_alpha'])
            ax.legend(loc='upper left')
        
        axes[-1].set_xlabel('Date', fontsize=self.style_config['font_size']['label'])
        fig.suptitle('Multi-Timeframe Analysis', 
                    fontsize=self.style_config['font_size']['title'],
                    fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def create_confidence_analysis_plot(self,
                                      patterns_data: Dict[str, Any]) -> plt.Figure:
        """Create detailed confidence analysis visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, 
                                                     figsize=self.style_config['figure_size'])
        
        # 1. Pattern confidence distribution
        patterns = patterns_data.get('composite_patterns', [])
        confidences = [p.get('composite_confidence', 0) for p in patterns]
        
        if confidences:
            ax1.hist(confidences, bins=10, alpha=0.7, 
                    color=self.color_palette['primary'])
            ax1.set_xlabel('Confidence Level')
            ax1.set_ylabel('Count')
            ax1.set_title('Pattern Confidence Distribution')
            ax1.grid(True, alpha=self.style_config['grid_alpha'])
        
        # 2. Support vs Confidence scatter
        support_counts = [p.get('support_count', 0) for p in patterns]
        if confidences and support_counts:
            scatter = ax2.scatter(support_counts, confidences,
                                s=100, alpha=0.6,
                                c=confidences, cmap='RdYlGn')
            ax2.set_xlabel('Support Count')
            ax2.set_ylabel('Confidence')
            ax2.set_title('Support vs Confidence')
            ax2.grid(True, alpha=self.style_config['grid_alpha'])
            plt.colorbar(scatter, ax=ax2)
        
        # 3. Timeframe confidence comparison
        patterns_by_tf = patterns_data.get('patterns_by_timeframe', {})
        if patterns_by_tf:
            timeframes = list(patterns_by_tf.keys())
            avg_confidences = []
            
            for tf in timeframes:
                tf_patterns = patterns_by_tf[tf].get('multiple_patterns', [])
                if tf_patterns:
                    avg_conf = np.mean([p.get('confidence', 0) for p in tf_patterns])
                    avg_confidences.append(avg_conf)
                else:
                    avg_confidences.append(0)
            
            bars = ax3.bar(timeframes, avg_confidences)
            for bar, conf in zip(bars, avg_confidences):
                bar.set_color(self._get_confidence_color(conf))
            
            ax3.set_xlabel('Timeframe')
            ax3.set_ylabel('Average Confidence')
            ax3.set_title('Confidence by Timeframe')
            ax3.grid(True, alpha=self.style_config['grid_alpha'], axis='y')
        
        # 4. Relationship quality
        relationships = patterns_data.get('pattern_relationships', {})
        if relationships:
            rel_types = ['Confirmations', 'Conflicts', 'Nested', 'Convergences']
            rel_counts = [
                len(relationships.get('confirmations', [])),
                len(relationships.get('conflicts', [])),
                len(relationships.get('nested_patterns', [])),
                len(relationships.get('convergences', []))
            ]
            
            colors = [self.color_palette['success'],
                     self.color_palette['danger'],
                     self.color_palette['warning'],
                     self.color_palette['primary']]
            
            ax4.pie(rel_counts, labels=rel_types, colors=colors,
                   autopct='%1.0f%%', startangle=90)
            ax4.set_title('Pattern Relationships')
        
        fig.suptitle('Confidence Analysis Dashboard',
                    fontsize=self.style_config['font_size']['title'],
                    fontweight='bold')
        plt.tight_layout()
        
        return fig 