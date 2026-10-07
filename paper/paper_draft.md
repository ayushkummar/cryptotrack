# CryptoTrack: Probabilistic Multi-Source Cryptocurrency Prediction and Cost-Aware Portfolio Optimization

## Abstract
Cryptocurrency portfolio construction is challenging because highly volatile asset returns can change rapidly across market regimes. The baseline study considered in this work combines historical trading data, Twitter sentiment and Google Trends with Support Vector Machine (SVM) price-movement prediction and a forecast-aware minimum-variance portfolio. CryptoTrack extends this framework by replacing hard binary movement signals with probabilistic forecasts, separating validation from final out-of-sample testing through walk-forward evaluation, and explicitly incorporating turnover and transaction costs into portfolio evaluation. The study compares market-only and multi-source prediction models, binary and probabilistic forecasts, conventional portfolio benchmarks and the proposed probability-aware portfolio strategy. Numerical results will be inserted only after the experimental pipeline is executed.

**Keywords:** cryptocurrency, probabilistic forecasting, sentiment analysis, SVM, portfolio optimization, walk-forward validation, transaction costs

## 1. Introduction
The baseline paper argues that historical asset returns alone may not adequately capture future cryptocurrency characteristics and therefore incorporates prediction information into portfolio construction. It uses historical trading data, Twitter sentiment and Google Trends, applies VADER to tweets, forecasts price movement using SVM, and combines the forecasts with a global minimum variance framework.

CryptoTrack focuses on three methodological extensions. First, binary movement predictions are replaced with probabilistic forecasts so that model confidence can be retained for portfolio allocation. Second, model development uses temporally ordered walk-forward validation with a separate final test period. Third, portfolio evaluation incorporates turnover and transaction costs, allowing gross and net performance to be distinguished.

## 2. Baseline Methodology
### 2.1 Data
The baseline paper uses open, high, low, close and volume data, Twitter sentiment and daily Google Trends. It studies Bitcoin, Ethereum, Ripple, Cardano, Dogecoin, Polkadot and Litecoin.

### 2.2 Sentiment
The baseline uses VADER to classify tweets into positive, neutral and negative sentiment and constructs daily sentiment indicators.

### 2.3 Forecasting
For horizons of one, three and five days, the baseline predicts whether the future closing price is above the current closing price using SVM.

### 2.4 Portfolio
The baseline incorporates the binary forecast vector into a minimum-variance portfolio model and compares it with minimum variance, maximum Sharpe, equal-weight and CRIX benchmarks.

## 3. Research Gap
1. Binary forecasts discard confidence information.
2. The baseline reports hyperparameter tuning according to test-set results, creating a risk of final-test contamination.
3. Portfolio performance should account for turnover and transaction costs.
4. Predictive metrics and economic portfolio performance should be evaluated separately.
5. A longer and rolling evaluation can test robustness across market regimes.

## 4. Proposed CryptoTrack Method
Market features are generated from lagged returns, volatility, moving-average relationships, volume changes and technical indicators. External sentiment and attention features are aligned by timestamp when available. The SVM produces a class prediction and probability of positive future movement.

The portfolio module combines the probability signal with covariance-based risk under long-only and maximum-weight constraints. At every rebalance, portfolio turnover is recorded and transaction costs are deducted from gross returns.

## 5. Experimental Design
E1: Market-only baseline.
E2: Market + sentiment.
E3: Market + Google Trends.
E4: Market + sentiment + Google Trends.
E5: Binary vs probabilistic SVM.
E6: Portfolio benchmark comparison.
E7: Transaction-cost sensitivity.
E8: Robustness across prediction horizons and time periods.

## 6. Results
This section will be completed only from generated experiments.

### 6.1 Prediction results
### 6.2 Portfolio results
### 6.3 Cost sensitivity
### 6.4 Robustness

## 7. Discussion
The discussion will distinguish classification performance from portfolio performance and will explicitly discuss uncertainty, transaction costs, turnover and possible failure regimes.

## 8. Limitations
Data-source availability, sentiment coverage, model assumptions, transaction-cost assumptions and the limitations of historical backtesting will be reported.

## 9. Conclusion
The conclusion will be written after the experiments and will report which hypotheses are supported by the observed results.
