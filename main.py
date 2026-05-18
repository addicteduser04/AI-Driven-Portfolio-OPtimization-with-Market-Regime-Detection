from Data_Extraction import build_data_pipeline
from Data_Train_HMM import train_and_decode_hmm
from Dynamic_asset_allocation import run_dynamic_allocation
from Data_Visualisation import plot_strategy_performance


if __name__ == "__main__":
    # 1. Build the data pipeline.
    market_data = build_data_pipeline()
    print("\n--- Pipeline Execution Complete ---")
    print(f"Total Trading Days Processed: {market_data.shape[0]}")
    print("\nFirst 5 rows of the structured dataset:")
    print(market_data.head())

    # 2. Train the HMM and decode regimes.
    market_data_with_regimes, trained_model = train_and_decode_hmm(market_data)

    # 3. Run the Markowitz + Kelly backtest.
    final_portfolio_data, performance_summary = run_dynamic_allocation(market_data_with_regimes)

    # 4. Generate the visual report.
    output_files = plot_strategy_performance(final_portfolio_data, performance_summary)
    print("\nSaved report files:")
    for output_file in output_files:
        print(f"- {output_file}")
