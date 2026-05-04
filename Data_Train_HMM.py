from hmmlearn.hmm import GaussianHMM
import pandas as pd
import numpy as np

def train_and_decode_hmm(dataset, n_states=2):
    """
    Trains a Gaussian HMM on market data and decodes the hidden regimes.
    """
    print(f"Training {n_states}-State Hidden Markov Model...")
    
    # 1. Prepare the Training Data (The "Emissions")
    # The HMM needs a 2D array of our observable features.
    X = dataset[['SPY_Log_Return', 'SPY_Realized_Vol']].values
    
    # 2. Initialize the Model
    # covariance_type='full' means the model assumes returns and volatility can interact
    # n_iter=100 gives the EM algorithm up to 100 loops to find the best mathematical fit
    hmm_model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100, random_state=42)
    
    # 3. Train the Model (Expectation-Maximization)
    hmm_model.fit(X)
    
    # 4. Decode the States (Viterbi Algorithm)
    # This assigns an integer (0 or 1) to every single day in the dataset
    hidden_states = hmm_model.predict(X)
    
    # Add the hidden states back to our pandas dataframe
    dataset['HMM_State'] = hidden_states
    
    # 5. Physically Label the Regimes (Interpretation)
    # The HMM assigns 0 and 1 randomly. We need to figure out which is the "Bear" market.
    # We do this by finding which state has the highest average realized volatility.
    
    state_volatility = {}
    for i in range(n_states):
        # Filter the dataset for each state and calculate the mean volatility
        mean_vol = dataset[dataset['HMM_State'] == i]['SPY_Realized_Vol'].mean()
        state_volatility[i] = mean_vol
        
    # The state with the maximum mean volatility is our "Risk-Off" (Bear) regime
    risk_off_state = max(state_volatility, key=state_volatility.get)
    
    # Create a human-readable column for your portfolio optimizer to use later
    dataset['Regime'] = np.where(dataset['HMM_State'] == risk_off_state, 'Risk-Off', 'Risk-On')
    
    print("\n--- HMM Training Complete ---")
    print(f"Risk-Off State identified as State {risk_off_state}")
    
    # Print the transition matrix (probability of switching states tomorrow)
    print("\nTransition Matrix:")
    print(np.round(hmm_model.transmat_, 3))
    
    return dataset, hmm_model

# ==========================================
# Assuming 'market_data' is the output from Phase 1
# market_data_with_regimes, trained_model = train_and_decode_hmm(market_data)
# ==========================================