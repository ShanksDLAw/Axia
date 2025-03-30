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
            
            # Handle special case for 'CASH' which doesn't need to be in returns data
            cash_weight = 0.0
            if 'CASH' in weights_series.index:
                cash_weight = weights_series['CASH']
                weights_series = weights_series.drop('CASH')
                
            if weights_series.empty and cash_weight > 0:
                # Portfolio is 100% cash, return zero-risk metrics
                return {
                    'Total Return': (1 + risk_free_rate) ** (len(self.returns) / 252) - 1,
                    'Annualized Return': risk_free_rate,
                    'Annualized Volatility': 0.0,
                    'Sharpe Ratio': 0.0,
                    'Sortino Ratio': 0.0,
                    'Max Drawdown': 0.0,
                    'Sector Allocation': {'Cash': 1.0},
                    'Number of Assets': 1,
                    'Cumulative Turnover': 0.0,
                    'Average Daily Turnover': 0.0
                }
            
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
            
            # Add back cash weight if present
            if cash_weight > 0:
                # Adjust other weights proportionally
                non_cash_sum = aligned_weights.sum()
                if non_cash_sum > 0:
                    aligned_weights = aligned_weights * (1 - cash_weight) / non_cash_sum
            
            # Normalize weights with validation
            if not np.isclose(aligned_weights.sum(), 1.0 - cash_weight, rtol=1e-3):
                if aligned_weights.sum() <= 0:
                    if cash_weight > 0:
                        # All in cash
                        return {
                            'Total Return': (1 + risk_free_rate) ** (len(self.returns) / 252) - 1,
                            'Annualized Return': risk_free_rate,
                            'Annualized Volatility': 0.0,
                            'Sharpe Ratio': 0.0,
                            'Sortino Ratio': 0.0,
                            'Max Drawdown': 0.0,
                            'Sector Allocation': {'Cash': 1.0},
                            'Number of Assets': 1,
                            'Cumulative Turnover': 0.0,
                            'Average Daily Turnover': 0.0
                        }
                    else:
                        raise ValueError("Sum of weights must be positive")
                aligned_weights = aligned_weights / aligned_weights.sum() * (1 - cash_weight)
                logging.info(f"Weights normalized. Original sum: {weights_series[common_assets].sum():.4f}")
            
            # Validate returns data
            if aligned_returns.isnull().any().any():
                logging.warning("NaN values found in returns data. Filling with 0.")
                aligned_returns = aligned_returns.fillna(0)
            
            # Calculate portfolio returns with validated data
            # Initialize portfolio returns and value tracking
            portfolio_returns = pd.Series(index=aligned_returns.index, dtype=float)
            portfolio_value = pd.Series(index=aligned_returns.index, dtype=float)
            portfolio_value.iloc[0] = 1.0
            
            # Initialize portfolio weights and tracking variables
            portfolio_weights = aligned_weights.copy()
            cumulative_turnover = 0.0
            last_rebalance_index = 0
            
            # Add cash component to portfolio weights if needed
            if cash_weight > 0:
                # For calculation purposes, we track cash separately
                cash_return = risk_free_rate / 252  # Daily risk-free rate
            
            for i in range(len(portfolio_returns)):
                try:
                    # Get current day's returns and handle missing values
                    returns_vector = aligned_returns.iloc[i].fillna(0)
                    
                    # Calculate portfolio return before costs (including cash component)
                    daily_return = returns_vector.dot(portfolio_weights)
                    if cash_weight > 0:
                        daily_return = daily_return * (1 - cash_weight) + cash_weight * cash_return
                    
                    # Update weights due to price changes (drift)
                    if portfolio_weights.sum() > 0:  # Only update if we have non-cash investments
                        drifted_weights = portfolio_weights * (1 + returns_vector)
                        drifted_sum = drifted_weights.sum()
                        
                        # Validate drifted weights
                        if drifted_sum > 0 and not np.isnan(drifted_sum):
                            drifted_weights = drifted_weights / drifted_sum * (1 - cash_weight)
                        else:
                            drifted_weights = portfolio_weights.copy()
                            logging.warning(f"Weight drift calculation failed at index {i}. Maintaining previous weights.")
                        
                        # Check for rebalancing (monthly)
                        days_since_last_rebalance = i - last_rebalance_index
                        is_rebalancing_day = days_since_last_rebalance >= 21
                        
                        # Calculate transaction costs and update weights
                        rebalancing_cost = 0.0
                        if is_rebalancing_day:
                            # Calculate turnover and costs
                            turnover = (aligned_weights - drifted_weights).abs().sum()
                            rebalancing_cost = transaction_cost * turnover
                            cumulative_turnover += turnover
                            
                            # Update weights and tracking
                            portfolio_weights = aligned_weights.copy()
                            last_rebalance_index = i
                        else:
                            portfolio_weights = drifted_weights
                    
                    # Calculate portfolio value and returns
                    if i == 0:
                        portfolio_value.iloc[i] = 1.0
                        portfolio_returns.iloc[i] = daily_return - rebalancing_cost
                    else:
                        # Calculate daily return with transaction costs
                        net_return = daily_return - rebalancing_cost
                        portfolio_value.iloc[i] = portfolio_value.iloc[i-1] * (1 + net_return)
                        portfolio_returns.iloc[i] = net_return  # Store actual daily returns with costs
                    
                except Exception as inner_e:
                    logging.error(f"Error at iteration {i}: {str(inner_e)}")
                    # Use previous return or zero, but don't crash the backtest
                    if i > 0:
                        portfolio_returns.iloc[i] = portfolio_returns.iloc[i-1]
                        portfolio_value.iloc[i] = portfolio_value.iloc[i-1]
                    else:
                        portfolio_returns.iloc[i] = 0
                        portfolio_value.iloc[i] = 1.0
            
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
            # Handle empty or invalid portfolio returns
            if portfolio_returns.empty:
                return {
                    'Total Return': 0.0,
                    'Annualized Return': 0.0,
                    'Annualized Volatility': 0.0,
                    'Sharpe Ratio': 0.0,
                    'Sortino Ratio': 0.0,
                    'Max Drawdown': 0.0
                }
                
            # Check for NaN values and handle them
            if portfolio_returns.isnull().any():
                logging.warning("NaN values found in portfolio returns. Filling with 0.")
                portfolio_returns = portfolio_returns.fillna(0)
                
            # Basic return metrics with validation
            try:
                total_return = (1 + portfolio_returns).prod() - 1
            except Exception as e:
                logging.error(f"Error calculating total return: {str(e)}")
                total_return = 0.0
                
            try:
                ann_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1 if len(portfolio_returns) > 0 else 0.0
            except Exception as e:
                logging.error(f"Error calculating annualized return: {str(e)}")
                ann_return = 0.0
            
            # Risk metrics with validation
            try:
                daily_vol = portfolio_returns.std()
                ann_vol = daily_vol * np.sqrt(252)
            except Exception as e:
                logging.error(f"Error calculating volatility: {str(e)}")
                daily_vol = 0.0
                ann_vol = 0.0
            
            # Maximum drawdown calculation with validation
            try:
                cum_returns = (1 + portfolio_returns).cumprod()
                rolling_max = cum_returns.expanding().max()
                drawdowns = cum_returns / rolling_max - 1
                max_drawdown = drawdowns.min()
            except Exception as e:
                logging.error(f"Error calculating drawdown: {str(e)}")
                max_drawdown = 0.0
            
            # Risk-adjusted metrics with proper annualization and validation
            try:
                daily_rf_rate = (1 + risk_free_rate) ** (1/252) - 1
                excess_returns = portfolio_returns - daily_rf_rate
                sharpe_ratio = np.sqrt(252) * excess_returns.mean() / daily_vol if daily_vol > 0 else 0
                downside_returns = portfolio_returns[portfolio_returns < daily_rf_rate]
                sortino_ratio = np.sqrt(252) * excess_returns.mean() / (downside_returns.std() or 1e-6)
            except Exception as e:
                logging.error(f"Error calculating risk-adjusted metrics: {str(e)}")
                sharpe_ratio = 0.0
                sortino_ratio = 0.0
            
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