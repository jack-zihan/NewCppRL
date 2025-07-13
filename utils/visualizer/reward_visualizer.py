import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, CheckButtons, Button
from matplotlib.patches import Rectangle, FancyBboxPatch
from pathlib import Path
import matplotlib.patches as mpatches
from datetime import datetime
import matplotlib.patheffects as path_effects
from matplotlib import cm
import seaborn as sns

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

NameMap = {
    'turn': 'turn',
    'frontier': 'Exploration',
    'weed': 'Weed Removal',
    'extra': 'Differential Diffusion Reward',
    'const': 'Time Penalty',
}

class RewardVisualizer:
    """Interactive reward trajectory visualizer with professional academic styling."""

    # Professional color palette inspired by Nature/Science publications 现在颜色可以了就这样吧
    COLORS = {
        'turn': '#0173B2',  # Bright blue
        'frontier': '#0173B2',  # Orange
        'weed': 'r',  # Purple
        'extra': '#029E73',  # Green
        'total': '#2E2E2E',  # Dark gray for total
        'const': '#DE8F05' # Wine red/Burgundy
    }
    # COLORS = {
    #     'turn': '#0173B2',  # Bright blue
    #     'frontier': '#E69F00',  # Orange
    #     'weed': '#D55E00',  # Purple
    #     'extra': '#56B4E9',  # Green
    #     'total': 'k',  # Dark gray for total
    #     'const': '#7F7F7F' # Wine red/Burgundy
    # }

    # Line styles for different rewards
    LINE_STYLES = {
        'turn': '-',
        'frontier': '-',
        'weed': '-',
        'extra': '-',
        'const': '--',  # 也可以用 '--' 虚线区分
        'total': '-',  # Solid line for total
    }

    # Markers for different rewards (optional)
    MARKERS = {
        'turn': 'o',
        'frontier': 's',
        'weed': '^',
        'extra': 'D',
        'const': 'v',
        'total': None,
    }


    def __init__(self, csv_path=None):
        """Initialize the visualizer."""
        self.csv_path = None
        self.df = None
        self.reward_types = ['turn', 'frontier', 'weed', 'extra', 'const']

        # Initialize states
        self.enabled = {r: True for r in self.reward_types}
        self.enabled['total'] = True
        self.coefficients = {r: 1.0 for r in self.reward_types}
        self.start_idx = 0
        self.end_idx = None
        self.use_markers = False
        self.smooth_curves = False
        self.show_cumulative = False  # New: cumulative rewards flag
        self.manual_y_range = False

        # Initialize UI elements that will be created later
        self.y_min_box = None
        self.y_max_box = None

        # Create the figure and layout
        self.create_layout()

        # Load data if provided
        if csv_path:
            self.load_data(csv_path)

    def create_layout(self):
        """Create the main figure layout with professional styling."""
        # Set up matplotlib parameters for publication quality
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'axes.linewidth': 1.2,
            'axes.grid': True,
            'grid.alpha': 0.15,
            'grid.linestyle': '-',
            'grid.linewidth': 0.8,
            'lines.linewidth': 2.5,
            'lines.antialiased': True,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'axes.edgecolor': '#333333',
            'text.color': '#333333',
            'axes.labelcolor': '#333333',
            'xtick.color': '#333333',
            'ytick.color': '#333333',
            'legend.frameon': True,
            'legend.framealpha': 0.95,
            'legend.fancybox': True,
            'legend.shadow': False,
            'legend.edgecolor': '#CCCCCC',
        })

        # Create figure with golden ratio proportions
        self.fig = plt.figure(figsize=(15, 9), dpi=100)
        self.fig.patch.set_facecolor('#FFFFFF')

        # Create grid layout with better spacing
        gs = self.fig.add_gridspec(7, 5, hspace=0.5, wspace=0.4,
                                   left=0.04, right=0.96, top=0.94, bottom=0.06)

        # Main plot area
        self.ax_main = self.fig.add_subplot(gs[:, 2:])
        self.setup_main_plot()

        # Control panel with elegant background
        panel_bg = FancyBboxPatch((0.02, 0.02), 0.36, 0.92,
                                  boxstyle="round,pad=0.02",
                                  transform=self.fig.transFigure,
                                  facecolor='#F8F9FA',
                                  edgecolor='#DEE2E6',
                                  linewidth=1.5,
                                  zorder=-1)
        self.fig.patches.append(panel_bg)

        # Title for control panel
        self.fig.text(0.2, 0.92, 'Control Panel', fontsize=13, fontweight='bold',
                      ha='center', color='#2E2E2E')

        # File input section
        self.setup_file_input()

        # Range controls
        self.setup_range_controls()

        # Reward component controls
        self.setup_reward_controls()

        # Style options
        self.setup_style_options()

        # Save button
        self.setup_save_button()

        # Initialize empty plot
        self.lines = {}
        self.create_plot_lines()

        plt.show(block=False)

    def setup_main_plot(self):
        """Set up the main plot with professional styling."""
        self.ax_main.set_facecolor('#FFFFFF')
        self.ax_main.grid(True, which='major', alpha=0.15, linestyle='-', linewidth=0.8)
        self.ax_main.grid(True, which='minor', alpha=0.05, linestyle=':', linewidth=0.5)

        # Remove top and right spines for cleaner look
        self.ax_main.spines['top'].set_visible(False)
        self.ax_main.spines['right'].set_visible(False)
        self.ax_main.spines['left'].set_linewidth(1.2)
        self.ax_main.spines['bottom'].set_linewidth(1.2)

        # Set labels with better formatting
        self.ax_main.set_xlabel('Step', fontsize=13, fontweight='medium', labelpad=10)
        self.ax_main.set_ylabel('Reward Value', fontsize=13, fontweight='medium', labelpad=10)

        # Add title with padding
        title = self.ax_main.set_title('Reward Components Over Training',
                                       fontsize=15, fontweight='bold', pad=20)

        # Minor ticks for better precision reading
        self.ax_main.minorticks_on()

    def setup_file_input(self):
        """Set up file input section."""
        # File input label
        self.fig.text(0.06, 0.85, 'CSV File Path:', fontsize=11, fontweight='bold')

        # Create file textbox with better styling
        self.file_box = TextBox(plt.axes([0.06, 0.81, 0.26, 0.035]), '',
                                initial='Click to paste path...',
                                color='#FFFFFF',
                                hovercolor='#F0F0F0')
        self.file_box.on_submit(self.load_data)

        # Add hint text
        self.fig.text(0.06, 0.78, '(Click box and paste with Ctrl+V)',
                      fontsize=9, style='italic', color='#666666')

    def setup_range_controls(self):
        """Set up range control section."""
        # Range controls label
        self.fig.text(0.06, 0.72, 'Display Range:', fontsize=11, fontweight='bold')

        # Range textboxes
        self.start_box = TextBox(plt.axes([0.06, 0.68, 0.1, 0.035]), 'Start:',
                                 initial='0', color='#FFFFFF')
        self.end_box = TextBox(plt.axes([0.19, 0.68, 0.1, 0.035]), 'End:',
                               initial='', color='#FFFFFF')
        self.start_box.on_submit(self.update_range)
        self.end_box.on_submit(self.update_range)

    def setup_reward_controls(self):
        """Set up reward component controls."""
        # Section title
        self.fig.text(0.06, 0.61, 'Reward Components:', fontsize=11, fontweight='bold')

        # Checkboxes for individual rewards
        check_ax = plt.axes([0.06, 0.38, 0.12, 0.21])
        self.check_rewards = CheckButtons(check_ax, self.reward_types,
                                          [self.enabled[r] for r in self.reward_types])

        # Color the checkbox labels
        for i, (label, reward) in enumerate(zip(self.check_rewards.labels, self.reward_types)):
            label.set_color(self.COLORS[reward])
            label.set_fontweight('medium')
            label.set_fontsize(10)

        self.check_rewards.on_clicked(self.toggle_reward)

        # Total reward checkbox
        total_ax = plt.axes([0.06, 0.33, 0.12, 0.05])  # 向下移动
        self.check_total = CheckButtons(total_ax, ['Total'], [self.enabled['total']])
        self.check_total.labels[0].set_color(self.COLORS['total'])
        self.check_total.labels[0].set_fontweight('bold')
        self.check_total.labels[0].set_fontsize(11)
        self.check_total.on_clicked(lambda x: self.toggle_total())

        # Coefficient controls
        self.fig.text(0.2, 0.58, 'Coefficients:', fontsize=10, fontweight='bold')

        self.coef_boxes = {}
        y_positions = [0.54, 0.49, 0.44, 0.39, 0.34]

        for i, reward in enumerate(self.reward_types):
            # Coefficient textbox
            box = TextBox(plt.axes([0.2, y_positions[i], 0.08, 0.035]), '',
                          initial=str(self.coefficients[reward]),
                          color='#FFFFFF')
            box.on_submit(lambda x, r=reward: self.update_coefficient(r, x))
            self.coef_boxes[reward] = box

            # Add coefficient symbol
            self.fig.text(0.29, y_positions[i] + 0.01, f'×{reward[0].upper()}',
                          fontsize=9, color=self.COLORS[reward])

    def setup_style_options(self):
        """Set up style options."""
        # Style options title
        self.fig.text(0.06, 0.27, 'Style Options:', fontsize=11, fontweight='bold')

        # Style checkboxes - Modified to include Cumulative Rewards
        style_ax = plt.axes([0.06, 0.17, 0.26, 0.09])
        self.style_checks = CheckButtons(style_ax,
                                         ['Show Markers', 'Smooth Curves', 'Cumulative Rewards'],
                                         [self.use_markers, self.smooth_curves, self.show_cumulative])

        # Color the cumulative checkbox differently to make it stand out
        self.style_checks.labels[2].set_color('#B8336A')
        self.style_checks.labels[2].set_fontweight('bold')

        self.style_checks.on_clicked(self.toggle_style)

        # Add Y-axis range controls - moved down slightly
        self.fig.text(0.06, 0.13, 'Y-axis Range:', fontsize=10, fontweight='bold')

        # Y-axis min/max boxes
        self.y_min_box = TextBox(plt.axes([0.06, 0.09, 0.08, 0.03]), 'Min:',
                                 initial='Auto', color='#FFFFFF')
        self.y_max_box = TextBox(plt.axes([0.16, 0.09, 0.08, 0.03]), 'Max:',
                                 initial='Auto', color='#FFFFFF')
        self.y_min_box.on_submit(self.update_y_range)
        self.y_max_box.on_submit(self.update_y_range)

        # Auto-fit button
        self.auto_fit_button = Button(plt.axes([0.25, 0.09, 0.07, 0.03]),
                                      'Auto Fit',
                                      color='#6A994E',
                                      hovercolor='#4A7C3E')
        self.auto_fit_button.label.set_color('white')
        self.auto_fit_button.label.set_fontsize(9)
        self.auto_fit_button.on_clicked(self.auto_fit_y_axis)

    def setup_save_button(self):
        """Set up save button."""
        self.save_button = Button(plt.axes([0.06, 0.04, 0.26, 0.045]),
                                  'Export Publication Figure',
                                  color='#0173B2',
                                  hovercolor='#0056A0')
        self.save_button.label.set_color('white')
        self.save_button.label.set_fontweight('bold')
        self.save_button.on_clicked(self.save_figure)

        # Add format options
        self.fig.text(0.06, 0.01, 'Export formats: PNG (300dpi), PDF, SVG',
                      fontsize=9, style='italic', color='#666666')

    def create_plot_lines(self):
        """Create plot lines with professional styling."""
        # Individual reward lines
        for reward in self.reward_types:
            line, = self.ax_main.plot([], [],
                                      label=NameMap[reward],
                                      color=self.COLORS[reward],
                                      linestyle=self.LINE_STYLES[reward],
                                      linewidth=7,#2.5,
                                      alpha=1,
                                      marker=self.MARKERS[reward] if self.use_markers else None,
                                      markersize=6,
                                      markevery=50)
            self.lines[reward] = line

        # Total reward line (thicker and different style)
        line, = self.ax_main.plot([], [],
                                  label='Total Reward',
                                  color=self.COLORS['total'],
                                  linestyle=self.LINE_STYLES['total'],
                                  linewidth=8,#3.5,
                                  alpha=1,
                                  zorder=10)  # Draw on top
        self.lines['total'] = line

        # Add legend with custom styling
        self.update_legend()

    def update_legend(self):
        """Update legend with professional styling."""
        handles = []
        labels = []

        for reward in self.reward_types + ['total']:
            if reward in self.lines and self.lines[reward].get_visible():
                handles.append(self.lines[reward])
                if reward == 'total':
                    label = 'Total Reward'
                else:
                    label = NameMap[reward]

                # Add cumulative indicator if enabled
                # if self.show_cumulative:
                #     label += ' [Cumulative]'

                labels.append(label)

        if handles:
            self.legend = self.ax_main.legend(handles, labels,
                                              loc='best',
                                              frameon=True,
                                              fancybox=True,
                                              framealpha=0.95,
                                              edgecolor='#CCCCCC',
                                              borderpad=1,
                                              columnspacing=2,
                                              handlelength=2.5)

    def load_data(self, csv_path):
        """Load data from CSV file."""
        try:
            csv_path = Path(csv_path.strip())
            if not csv_path.exists():
                self.show_message(f"File not found: {csv_path}", error=True)
                return

            self.df = pd.read_csv(csv_path)
            self.csv_path = csv_path

            # Validate required columns
            required_cols = ['step'] + self.reward_types
            missing_cols = [col for col in required_cols if col not in self.df.columns]
            if missing_cols:
                self.show_message(f"Missing columns: {missing_cols}", error=True)
                return

            # Update range
            self.start_idx = 0
            self.end_idx = len(self.df) - 1
            self.start_box.set_val(str(self.start_idx))
            self.end_box.set_val(str(self.end_idx))

            # Update plot
            self.update_plot()
            self.show_message(f"Loaded: {csv_path.name} ({len(self.df)} steps)")

        except Exception as e:
            self.show_message(f"Error loading file: {str(e)}", error=True)

    def update_range(self, text):
        """Update the display range."""
        if self.df is None:
            return

        try:
            self.start_idx = max(0, int(self.start_box.text))
            end_text = self.end_box.text.strip()
            self.end_idx = int(end_text) if end_text else len(self.df) - 1
            self.end_idx = min(self.end_idx, len(self.df) - 1)

            self.update_plot()
        except ValueError:
            pass

    def toggle_reward(self, label):
        """Toggle individual reward visibility."""
        self.enabled[label] = not self.enabled[label]
        self.update_plot()

    def toggle_total(self):
        """Toggle total reward visibility."""
        self.enabled['total'] = not self.enabled['total']
        self.update_plot()

    def toggle_style(self, label):
        """Toggle style options."""
        if label == 'Show Markers':
            self.use_markers = not self.use_markers
        elif label == 'Smooth Curves':
            self.smooth_curves = not self.smooth_curves
        elif label == 'Cumulative Rewards':
            self.show_cumulative = not self.show_cumulative
            # Update Y-axis label when toggling cumulative
            if self.show_cumulative:
                self.ax_main.set_ylabel('Cumulative Reward Value', fontsize=13, fontweight='medium', labelpad=10)
                self.ax_main.set_title('Cumulative Reward Components Over Training',
                                       fontsize=15, fontweight='bold', pad=20)
            else:
                self.ax_main.set_ylabel('Reward Value', fontsize=13, fontweight='medium', labelpad=10)
                self.ax_main.set_title('Reward Components Over Training',
                                       fontsize=15, fontweight='bold', pad=20)

        if label in ['Show Markers', 'Smooth Curves']:
            self.recreate_lines()

        self.update_plot()

    def recreate_lines(self):
        """Recreate lines with new style settings."""
        # Remove old lines
        for line in self.lines.values():
            line.remove()
        self.lines.clear()

        # Create new lines
        self.create_plot_lines()

    def update_coefficient(self, reward, value):
        """Update coefficient value."""
        try:
            self.coefficients[reward] = float(value)
            self.update_plot()
        except ValueError:
            pass

    def smooth_data(self, data, window=5):
        """Apply smoothing to data if enabled."""
        if not self.smooth_curves or len(data) < window:
            return data

        # Use pandas rolling mean for smoothing
        smoothed = pd.Series(data).rolling(window=window, center=True).mean()
        # Fill NaN values at edges
        smoothed = smoothed.fillna(method='bfill').fillna(method='ffill')
        return smoothed.values

    def update_y_range(self, text):
        """Update Y-axis range manually."""
        try:
            y_min_text = self.y_min_box.text.strip().lower()
            y_max_text = self.y_max_box.text.strip().lower()

            if y_min_text == 'auto' or y_max_text == 'auto':
                # If either is auto, use automatic scaling
                self.update_plot()
            else:
                # Manual range
                y_min = float(y_min_text)
                y_max = float(y_max_text)
                if y_min < y_max:
                    self.ax_main.set_ylim(y_min, y_max)
                    self.fig.canvas.draw_idle()
        except ValueError:
            # Invalid input, revert to auto
            self.update_plot()

    def auto_fit_y_axis(self, event):
        """Auto-fit Y-axis to visible data."""
        self.y_min_box.set_val('Auto')
        self.y_max_box.set_val('Auto')
        self.update_plot()
        self.show_message("Y-axis auto-fitted to data")

    def update_plot(self):
        """Update the plot with current settings."""
        if self.df is None:
            return

        # Get data slice
        data_slice = self.df.iloc[self.start_idx:self.end_idx + 1]
        steps = data_slice['step'].values

        # Collect all visible data for proper scaling
        all_visible_data = []

        # Update individual reward lines
        for reward in self.reward_types:
            if self.enabled[reward]:
                values = data_slice[reward].values

                # Apply coefficient to the values
                values = values * self.coefficients[reward]

                # Calculate cumulative if enabled
                if self.show_cumulative:
                    # Get cumulative values from the beginning up to the current range
                    # Apply coefficient before cumsum
                    cumulative_values = (
                                self.df[reward].iloc[:self.end_idx + 1] * self.coefficients[reward]).cumsum().values
                    values = cumulative_values[self.start_idx:self.end_idx + 1]

                if self.smooth_curves:
                    values = self.smooth_data(values)

                self.lines[reward].set_data(steps, values)
                self.lines[reward].set_visible(True)
                all_visible_data.extend(values)
            else:
                self.lines[reward].set_visible(False)

        # Update total line
        if self.enabled['total']:
            # Calculate weighted sum
            total_values = np.zeros(len(self.df))  # Start with full dataframe
            for reward in self.reward_types:
                if reward in self.df.columns:
                    reward_values = self.df[reward].values
                    total_values += reward_values * self.coefficients[reward]

            # Get cumulative if enabled
            if self.show_cumulative:
                total_values = np.cumsum(total_values)

            # Slice to display range
            total_values_slice = total_values[self.start_idx:self.end_idx + 1]

            if self.smooth_curves:
                total_values_slice = self.smooth_data(total_values_slice, window=10)

            self.lines['total'].set_data(steps, total_values_slice)
            self.lines['total'].set_visible(True)
            all_visible_data.extend(total_values_slice)
        else:
            self.lines['total'].set_visible(False)

        # Update legend
        self.update_legend()

        # Smart axis limits adjustment
        if all_visible_data:
            # Only auto-adjust if not using manual range
            if self.y_min_box is not None and self.y_max_box is not None:
                y_min_text = self.y_min_box.text.strip().lower()
                y_max_text = self.y_max_box.text.strip().lower()

                if y_min_text == 'auto' or y_max_text == 'auto':
                    # Calculate actual data range
                    data_min = np.min(all_visible_data)
                    data_max = np.max(all_visible_data)
                    data_range = data_max - data_min

                    # Handle edge cases
                    if data_range < 1e-6:  # Nearly constant data
                        data_range = max(abs(data_min), abs(data_max), 1.0) * 0.1

                    # Add reasonable padding (10% of range)
                    padding = data_range * 0.1
                    self.ax_main.set_ylim(data_min - padding, data_max + padding)
            else:
                # Initial load, use auto scaling
                data_min = np.min(all_visible_data)
                data_max = np.max(all_visible_data)
                data_range = data_max - data_min

                if data_range < 1e-6:
                    data_range = max(abs(data_min), abs(data_max), 1.0) * 0.1

                padding = data_range * 0.1
                self.ax_main.set_ylim(data_min - padding, data_max + padding)

            # Set x-axis limits
            if len(steps) > 0:
                self.ax_main.set_xlim(steps[0], steps[-1])
        else:
            # No visible data - reset to default view
            self.ax_main.set_ylim(-1, 1)
            self.ax_main.set_xlim(0, 1)

        # Redraw
        self.fig.canvas.draw_idle()

    def save_figure(self, event):
        """Save the current figure in multiple formats."""
        if self.csv_path is None:
            self.show_message("No data loaded to save.", error=True)
            return

        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cumulative_suffix = '_cumulative' if self.show_cumulative else ''
        base_name = f'reward_trajectory{cumulative_suffix}_{timestamp}'

        # Create a new figure for saving
        save_fig, save_ax = plt.subplots(figsize=(10, 6), dpi=300)

        # Copy plot styling
        save_ax.set_facecolor('#FFFFFF')
        save_ax.grid(True, which='major', alpha=0.15, linestyle='-', linewidth=0.8)
        save_ax.grid(True, which='minor', alpha=0.05, linestyle=':', linewidth=0.5)
        save_ax.spines['top'].set_visible(False)
        save_ax.spines['right'].set_visible(False)
        save_ax.spines['left'].set_linewidth(1.2)
        save_ax.spines['bottom'].set_linewidth(1.2)
        save_ax.minorticks_on()

        # Copy current plot data
        handles = []
        labels = []

        for reward in self.reward_types + ['total']:
            if reward in self.lines and self.lines[reward].get_visible():
                xdata, ydata = self.lines[reward].get_data()
                if reward == 'total':
                    label = 'Total Reward'
                    line, = save_ax.plot(xdata, ydata,
                                         label=label,
                                         color=self.COLORS[reward],
                                         linestyle=self.LINE_STYLES[reward],
                                         linewidth=7,
                                         alpha=1,
                                         zorder=10)
                else:
                    label = NameMap[reward]
                    line, = save_ax.plot(xdata, ydata,
                                         label=label,
                                         color=self.COLORS[reward],
                                         linestyle=self.LINE_STYLES[reward],
                                         linewidth=8,
                                         alpha=1,
                                         marker=self.MARKERS[reward] if self.use_markers else None,
                                         markersize=6,
                                         markevery=50)

        save_ax.set_xlabel('Training Step', fontsize=13, fontweight='medium', labelpad=10)

        if self.show_cumulative:
            save_ax.set_ylabel('Cumulative Reward Value', fontsize=13, fontweight='medium', labelpad=10)
            save_ax.set_title('Cumulative Reward Components Over Training', fontsize=15, fontweight='bold', pad=20)
        else:
            save_ax.set_ylabel('Reward Value', fontsize=13, fontweight='medium', labelpad=10)
            save_ax.set_title('Reward Components Over Training', fontsize=15, fontweight='bold', pad=20)

        # Add legend
        legend = save_ax.legend(loc='best', frameon=True, fancybox=True,
                                framealpha=0.95, edgecolor='#CCCCCC',
                                borderpad=1, columnspacing=2, handlelength=2.5)

        # Copy axis limits
        save_ax.set_xlim(self.ax_main.get_xlim())
        save_ax.set_ylim(self.ax_main.get_ylim())

        # Save in multiple formats
        plt.tight_layout()

        # PNG (high resolution)
        png_path = self.csv_path.parent / f'{base_name}.png'
        save_fig.savefig(png_path, dpi=300, bbox_inches='tight',
                         facecolor='white', edgecolor='none')

        # PDF (vector format)
        pdf_path = self.csv_path.parent / f'{base_name}.pdf'
        save_fig.savefig(pdf_path, format='pdf', bbox_inches='tight',
                         facecolor='white', edgecolor='none')

        # SVG (vector format)
        svg_path = self.csv_path.parent / f'{base_name}.svg'
        save_fig.savefig(svg_path, format='svg', bbox_inches='tight',
                         facecolor='white', edgecolor='none')

        plt.close(save_fig)

        self.show_message(f"Saved: PNG, PDF, SVG in {self.csv_path.parent}")

    def show_message(self, message, error=False):
        """Show a message in the title."""
        original_title = self.ax_main.get_title()
        color = '#CC0000' if error else '#009900'
        self.ax_main.set_title(f'{original_title} - {message}',
                               fontsize=14, fontweight='bold', color=color)
        self.fig.canvas.draw_idle()

        # Reset title after 3 seconds
        plt.pause(3)
        self.ax_main.set_title(original_title, fontsize=15, fontweight='bold', color='black')
        self.fig.canvas.draw_idle()


def main():
    """Main entry point."""
    print("=== Professional Reward Trajectory Visualizer ===")
    print("Publication-quality visualization for reinforcement learning\n")

    # Get CSV path from command line or prompt
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        print(f"Loading: {csv_path}")
    else:
        print("Usage: python visualizer.py [csv_path]")
        print("\nYou can also:")
        print("1. Paste the path directly in the UI (click the input box)")
        print("2. Use Ctrl+V (Windows/Linux) or Cmd+V (Mac) to paste")
        print("\nExpected CSV format:")
        print("- Columns: step, turn, frontier, weed, extra")
        print("- Each row represents a training step\n")
        csv_path = None

    # Create visualizer
    try:
        visualizer = RewardVisualizer(csv_path)

        print("\nFeatures:")
        print("✓ Individual reward component visualization")
        print("✓ Weighted total reward calculation")
        print("✓ Adjustable coefficients for each component")
        print("✓ Interactive range selection")
        print("✓ Export to PNG (300dpi), PDF, and SVG")
        print("✓ Optional markers and curve smoothing")
        print("✓ Cumulative reward visualization")
        print("\nClose the window to exit.")

        plt.show()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()