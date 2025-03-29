import pandas as pd
import numpy as np
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class Backtester:
    def __init__(self, price_data: pd.DataFrame, sectors: dict[str, str]):
        if price_data.empty:
            raise ValueError("Cannot initialize backtester with empty price data")
        self.price_data = price_data
        self.sectors = sectors
        self.returns = price_data.pct_change().dropna()
    
    def run_backtest(self, weights: dict[str, float], 
                    risk_free_rate: float = 0.03, 
                    transaction_cost: float = 0.0001) -> dict:
        """Run backtest with transaction costs and enhanced metrics"""
        try:
            # Create weights series and align with returns data
            weights_series = pd.Series(weights)
            common_assets = self.returns.columns.intersection(weights_series.index)
            
            # Enhanced validation for common assets
            if len(common_assets) == 0:
                missing_assets = set(weights_series.index) - set(self.returns.columns)
                available_assets = set(self.returns.columns)
                error_msg = f"No common assets between weights and returns data.\n"
                error_msg += f"Missing assets: {missing_assets}\n"
                error_msg += f"Available assets: {available_assets}"
                raise ValueError(error_msg)
                
            # Align and validate data with proper error handling
            aligned_returns = self.returns[common_assets]
            aligned_weights = weights_series[common_assets]
            
            # Normalize weights if needed
            if not np.isclose(aligned_weights.sum(), 1.0, rtol=1e-3):
                aligned_weights = aligned_weights / aligned_weights.sum()
                logging.warning(f"Weights were automatically normalized to sum to 1.0")
            
            if not np.isclose(aligned_weights.sum(), 1.0, rtol=1e-3):
                logging.warning(f"Portfolio weights sum to {aligned_weights.sum():.4f}, not 1.0")
            
            # Calculate portfolio returns with aligned data
            portfolio_returns = aligned_returns.dot(aligned_weights)
            
            # Calculate transaction costs (0.01% per trade)
            turnover = aligned_weights.abs().sum()  # Initial allocation
            portfolio_returns.iloc[0] -= transaction_cost * turnover
            
            # Ongoing turnover calculation with proper weight alignment
            for i in range(1, len(portfolio_returns)):
                # Calculate turnover based on aligned weights
                turnover = (aligned_weights - aligned_weights * (1 + aligned_returns.iloc[i-1])).abs().sum()
                portfolio_returns.iloc[i] -= transaction_cost * turnover
            
            # Calculate comprehensive metrics
            metrics = self._calculate_metrics(portfolio_returns, risk_free_rate)
            metrics['Sector Allocation'] = self._analyze_sectors({k: v for k, v in weights.items() if k in common_assets})
            
            return metrics
        except Exception as e:
            logging.error(f"Backtest failed: {str(e)}")
            raise
    
    def _analyze_sectors(self, weights: dict[str, float]) -> dict[str, float]:
        """Analyze sector allocation of the portfolio"""
        sectors = defaultdict(float)
        for symbol, weight in weights.items():
            sectors[self.sectors.get(symbol, 'Unknown')] += weight
        return dict(sectors)

    def _calculate_metrics(self, portfolio_returns: pd.Series, risk_free_rate: float = 0.03) -> dict:
        """Calculate comprehensive portfolio performance metrics"""
        try:
            # Basic return metrics
            total_return = (1 + portfolio_returns).prod() - 1
            ann_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
            
            # Risk metrics
            daily_vol = portfolio_returns.std()
            ann_vol = daily_vol * np.sqrt(252)
            
            # Maximum drawdown calculation
            cum_returns = (1 + portfolio_returns).cumprod()
            rolling_max = cum_returns.expanding().max()
            drawdowns = cum_returns / rolling_max - 1
            max_drawdown = drawdowns.min()
            
            # Risk-adjusted metrics with proper annualization
            daily_rf_rate = (1 + risk_free_rate) ** (1/252) - 1
            excess_returns = portfolio_returns - daily_rf_rate
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / daily_vol if daily_vol != 0 else 0
            downside_returns = portfolio_returns[portfolio_returns < daily_rf_rate]
            sortino_ratio = np.sqrt(252) * excess_returns.mean() / (downside_returns.std() or 1e-6)
            
            return {
                'Total Return': total_return,
                'Annualized Return': ann_return,
                'Annualized Volatility': ann_vol,
                'Sharpe Ratio': sharpe_ratio,
                'Sortino Ratio': sortino_ratio,
                'Max Drawdown': max_drawdown
            }
            
        except Exception as e:
            logging.error(f"Error calculating metrics: {str(e)}")
            return {
                'Total Return': 0.0,
                'Annualized Return': 0.0,
                'Annualized Volatility': 0.0,
                'Sharpe Ratio': 0.0,
                'Sortino Ratio': 0.0,
                'Max Drawdown': 0.0
            }