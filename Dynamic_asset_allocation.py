import pandas as pd
import numpy as np

def run_dynamic_allocation(dataset):
    """
    Simulates dynamic portfolio allocation based on HMM regime detection.
    """
    print("Initializing Multi-Factor Allocation Engine...")
    
    # 1. Convert Log Returns back to Simple Returns for Portfolio Math
    # Formula: Simple Return = exp(Log Return) - 1
    factor_columns = ['MTUM', 'VLUE', 'QUAL', 'USMV']
    simple_returns = np.exp(dataset[factor_columns]) - 1
    
    # 2. Define the Target Weights for Each Regime
    # Ensure each dictionary sums perfectly to 1.0 (100% of the portfolio)
    target_weights = {
        'Risk-On': {
            'MTUM': 0.70,  # Aggressive growth
            'QUAL': 0.15,
            'VLUE': 0.15,
            'USMV': 0.00   # No need for low volatility in a bull market
        },
        'Risk-Off': {
            'MTUM': 0.00,  # Dump momentum, it crashes hardest in a bear market
            'QUAL': 0.30,  # Companies with strong balance sheets
            'VLUE': 0.30,  # Undervalued assets
            'USMV': 0.40   # Heavy protection
        }
    }
    
    print("Applying dynamic weights based on HMM states...")
    
    # 3. Apply the Weights Day by Day
    # We create a list to store the daily portfolio return
    dynamic_portfolio_returns = []
    
    # For a real production system, you would vectorize this or use backtrader, 
    # but looping makes the logic clear for building the prototype
    for date, row in dataset.iterrows():
        current_regime = row['Regime']
        
        # Get the weights for today's AI-predicted regime
        w_MTUM = target_weights[current_regime]['MTUM']
        w_QUAL = target_weights[current_regime]['QUAL']
        w_VLUE = target_weights[current_regime]['VLUE']
        w_USMV = target_weights[current_regime]['USMV']
        
        # Calculate daily portfolio return (Dot product of weights and simple returns)
        # return = (w1 * r1) + (w2 * r2) + ...
        daily_ret = (
            w_MTUM * simple_returns.loc[date, 'MTUM'] +
            w_QUAL * simple_returns.loc[date, 'QUAL'] +
            w_VLUE * simple_returns.loc[date, 'VLUE'] +
            w_USMV * simple_returns.loc[date, 'USMV']
        )
        dynamic_portfolio_returns.append(daily_ret)
        
    # 4. Attach Results to Dataset
    dataset['Dynamic_Portfolio_Return'] = dynamic_portfolio_returns
    
    # 5. Create a Static Benchmark (Equal Weight) for Comparison
    # A standard "dumb" portfolio allocates 25% to all 4 factors permanently
    dataset['Static_Portfolio_Return'] = simple_returns.mean(axis=1) 
    
    # 6. Calculate Cumulative Growth ($1 invested)
    # Formula for compounding simple returns over time: product(1 + return)
    dataset['Dynamic_Cumulative'] = (1 + dataset['Dynamic_Portfolio_Return']).cumprod()
    dataset['Static_Cumulative'] = (1 + dataset['Static_Portfolio_Return']).cumprod()
    
    print("\n--- Allocation Complete ---")
    
    # Quick Performance Summary
    dyn_total_return = (dataset['Dynamic_Cumulative'].iloc[-1] - 1) * 100
    stat_total_return = (dataset['Static_Cumulative'].iloc[-1] - 1) * 100
    
    print(f"Static Benchmark Total Return:  {stat_total_return:.2f}%")
    print(f"AI Dynamic Portfolio Return:    {dyn_total_return:.2f}%")
    
    return dataset

# ==========================================
# Assuming 'market_data_with_regimes' is the output from Phase 2
# final_portfolio_data = run_dynamic_allocation(market_data_with_regimes)
# ==========================================