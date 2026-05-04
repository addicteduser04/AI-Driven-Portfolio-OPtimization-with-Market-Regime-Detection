import matplotlib.pyplot as plt
import pandas as pd

def plot_strategy_performance(dataset):
    """
    Plots the cumulative returns of the dynamic vs. static portfolios
    and shades the background to show the HMM's detected regimes.
    """
    print("Generating performance visualization...")
    
    # Set a dark theme for high contrast
    plt.style.use('dark_background')
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # 1. Plot the Equity Curves
    # Static benchmark in a muted gray
    ax.plot(dataset.index, dataset['Static_Cumulative'], 
            label='Static Benchmark (Equal Weight)', color='#888888', linewidth=1.5)
    
    # AI portfolio in a bold cyan
    ax.plot(dataset.index, dataset['Dynamic_Cumulative'], 
            label='AI Dynamic Portfolio (HMM)', color='#00ffcc', linewidth=2.5)
    
    # 2. Add Regime Shading (The "Risk-Off" zones)
    # We loop through the dataset and draw a semi-transparent red box 
    # over the days where the AI declared 'Risk-Off'
    in_risk_off = False
    start_date = None
    
    for date, row in dataset.iterrows():
        if row['Regime'] == 'Risk-Off' and not in_risk_off:
            # Regime just started
            start_date = date
            in_risk_off = True
        elif row['Regime'] == 'Risk-On' and in_risk_off:
            # Regime just ended, draw the shaded block
            ax.axvspan(start_date, date, color='#ff3366', alpha=0.2, lw=0)
            in_risk_off = False
            
    # Catch a case where the dataset ends while still in a Risk-Off regime
    if in_risk_off:
        ax.axvspan(start_date, dataset.index[-1], color='#ff3366', alpha=0.2, lw=0)
        
    # 3. Format the Chart
    ax.set_title('AI-Driven Multi-Factor Portfolio vs. Static Benchmark', 
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_ylabel('Cumulative Return (Growth of $1)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    
    # Add a subtle grid
    ax.grid(color='#333333', linestyle='--', alpha=0.7)
    
    # Place the legend in the top left
    ax.legend(loc='upper left', frameon=True, facecolor='#111111', edgecolor='#333333')
    
    # Clean up the layout and display
    plt.tight_layout()
    plt.savefig('strategy_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

# ==========================================
# Assuming 'final_portfolio_data' is the output from Phase 3
# plot_strategy_performance(final_portfolio_data)
# ==========================================