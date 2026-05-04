from Data_Extraction import build_data_pipeline
from Data_Train_HMM import train_and_decode_hmm
from Dynamic_asset_allocation import run_dynamic_allocation
from Data_Visualisation import plot_strategy_performance


if __name__ == "__main__":
    # 1. Build the Data Pipeline (Phase 1)
    
    market_data = build_data_pipeline()
    print("\n--- Pipeline Execution Complete ---")
    print(f"Total Trading Days Processed: {market_data.shape[0]}")
    print("\nFirst 5 rows of the structured dataset:")
    print(market_data.head())
    
    # 2. Train the HMM and Decode Regimes (Phase 2)
    
    market_data_with_regimes, trained_model = train_and_decode_hmm(market_data)
    
    # 3. Run the Dynamic Asset Allocation Backtest (Phase 3)
    
    final_portfolio_data = run_dynamic_allocation(market_data_with_regimes)
    
    # 4. Visualize the Performance (Phase 4)
    plot_strategy_performance(final_portfolio_data)
