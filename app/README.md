# CryptoTrack Dashboard

A Streamlit research dashboard for the CryptoTrack project.

## Install

From the project root:

```powershell
pip install -r app/requirements-app.txt
```

## Run

```powershell
streamlit run app/streamlit_app.py
```

The dashboard expects the existing project structure:

- `experiments/multi_source_portfolio_summary.csv`
- `experiments/prediction_experiment_results.csv`
- `experiments/final_regime_robustness.csv`
- `experiments/final_regime_sharpe_table.csv`
- `experiments/final_regime_return_table.csv`
- `experiments/final_regime_drawdown_table.csv`
- `experiments/final_regime_probability_vs_binary.csv`
- `experiments/transaction_cost_sensitivity.csv`
- `experiments/phi_sensitivity.csv`
- `backtests/*.csv`

The app does not modify your experiment files.
