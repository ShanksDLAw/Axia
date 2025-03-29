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
        """Run backtest with transaction costs and enhanced metrics with improved validation."""
        try:
            # Validate input weights
            if not weights:
                raise ValueError("Empty weights dictionary provided")
            
            # Filter out invalid weights and log warnings
            valid_weights = {}
            for symbol, weight in weights.items():
                if not isinstance(weight, (int, float)) or np.isnan(weight):
                    logging.warning(f"Invalid weight for {symbol}: {weight}. Skipping.")
                    continue
                if weight < 0:
                    logging.warning(f"Negative weight for {symbol}: {weight}. Setting to 0.")
                    continue
                valid_weights[symbol] = weight
            
            if not valid_weights:
                raise ValueError("No valid weights after filtering")
            
            # Create weights series and align with returns data
            weights_series = pd.Series(valid_weights)
            common_assets = self.returns.columns.intersection(weights_series.index)
            
            # Enhanced validation for common assets with detailed error message
            if len(common_assets) == 0:
                missing_assets = set(weights_series.index) - set(self.returns.columns)
                available_assets = set(self.returns.columns)
                error_msg = f"No common assets between weights and returns data.\n"
                error_msg += f"Missing assets: {missing_assets}\n"
                error_msg += f"Available assets: {available_assets}"
                raise ValueError(error_msg)
            
            if len(common_assets) < len(weights_series):
                excluded_assets = set(weights_series.index) - set(common_assets)
                logging.warning(f"Some assets were excluded due to missing data: {excluded_assets}")
            
            # Align and validate data with proper error handling
            aligned_returns = self.returns[common_assets]
            aligned_weights = weights_series[common_assets]
            
            # Normalize weights with validation
            if not np.isclose(aligned_weights.sum(), 1.0, rtol=1e-3):
                if aligned_weights.sum() <= 0:
                    raise ValueError("Sum of weights must be positive")
                aligned_weights = aligned_weights / aligned_weights.sum()
                logging.info(f"Weights normalized. Original sum: {weights_series[common_assets].sum():.4f}")
            
            # Validate returns data
            if aligned_returns.isnull().any().any():
                logging.warning("NaN values found in returns data. Filling with 0.")
                aligned_returns = aligned_returns.fillna(0)
            
            # Calculate portfolio returns with validated data
            portfolio_returns = aligned_returns.dot(aligned_weights)
            
            # Enhanced transaction cost handling
            initial_turnover = aligned_weights.abs().sum()
            portfolio_returns.iloc[0] -= transaction_cost * initial_turnover
            
            # Track portfolio evolution with improved weight drift handling
            portfolio_weights = aligned_weights.copy()
            cumulative_turnover = initial_turnover
            
            for i in range(1, len(portfolio_returns)):
                try:
                    # Calculate weight drift with validation
                    returns_vector = aligned_returns.iloc[i-1]
                    if returns_vector.isnull().any():
                        returns_vector = returns_vector.fillna(0)
                        logging.warning(f"NaN returns at index {i-1}. Filled with 0.")
                    
                    drifted_weights = portfolio_weights * (1 + returns_vector)
                    drifted_sum = drifted_weights.sum()
                    
                    if drifted_sum <= 0:
                        logging.warning(f"Invalid drifted weights sum at index {i}. Using previous weights.")
                        drifted_weights = portfolio_weights
                    else:
                        drifted_weights = drifted_weights / drifted_sum
                    
                    # Calculate and apply transaction costs
                    turnover = (aligned_weights - drifted_weights).abs().sum()
                    cumulative_turnover += turnover
                    portfolio_returns.iloc[i] -= transaction_cost * turnover
                    
                    # Update for next iteration
                    portfolio_weights = aligned_weights.copy()
                    
                except Exception as inner_e:
                    logging.error(f"Error at iteration {i}: {str(inner_e)}")
                    portfolio_returns.iloc[i] = 0
            
            # Calculate comprehensive metrics
            metrics = self._calculate_metrics(portfolio_returns, risk_free_rate)
            metrics.update({
                'Sector Allocation': self._analyze_sectors({k: v for k, v in weights.items() if k in common_assets}),
                'Number of Assets': len(common_assets),
                'Cumulative Turnover': cumulative_turnover,
                'Average Daily Turnover': cumulative_turnover / len(portfolio_returns)
            })
            
            return metrics
            
        except Exception as e:
            logging.error(f"Backtest failed: {str(e)}")
            return {
                'error': str(e),
                'Total Return': 0.0,
                'Annualized Return': 0.0,
                'Annualized Volatility': 0.0,
                'Sharpe Ratio': 0.0,
                'Sortino Ratio': 0.0,
                'Max Drawdown': 0.0,
                'Sector Allocation': {'Unknown': 1.0}
            }
    
    def _analyze_sectors(self, weights: dict[str, float]) -> dict[str, float]:
        """Analyze sector allocation of the portfolio with enhanced validation."""
        try:
            sectors = defaultdict(float)
            total_weight = 0.0
            
            # Process each symbol and accumulate sector weights
            for symbol, weight in weights.items():
                if not isinstance(weight, (int, float)) or np.isnan(weight):
                    logging.warning(f"Invalid weight for {symbol}: {weight}. Skipping.")
                    continue
                    
                sector = self.sectors.get(symbol)
                if sector is None:
                    logging.warning(f"No sector found for {symbol}. Allocating to 'Unknown'.")
                    sector = 'Unknown'
                
                sectors[sector] += weight
                total_weight += weight
            
            # Normalize sector weights if total weight is not close to 1.0
            if not np.isclose(total_weight, 1.0, rtol=1e-3) and total_weight > 0:
                for sector in sectors:
                    sectors[sector] /= total_weight
                logging.info(f"Sector weights normalized. Original total: {total_weight:.4f}")
            
            return dict(sectors)
        except Exception as e:
            logging.error(f"Error in sector analysis: {str(e)}")
            return {'Unknown': 1.0}

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