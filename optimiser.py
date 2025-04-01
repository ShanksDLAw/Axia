import numpy as np
import logging
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from pypfopt import risk_models, objective_functions
from pypfopt import EfficientFrontier

class PortfolioOptimizer:
    def __init__(self, price_data: pd.DataFrame, sectors: Optional[Dict[str, str]] = None):
        """Initialize the Portfolio Optimizer.

        Args:
            price_data: DataFrame of asset prices (each column represents an asset)
            sectors: Dictionary mapping asset symbols to their respective sectors
        """
        if not isinstance(price_data, pd.DataFrame):
            raise TypeError("price_data must be a pandas DataFrame")
        if price_data.empty:
            raise ValueError("price_data cannot be empty")
            
        self.price_data = price_data
        self.sectors = sectors
        self.returns = self._calculate_returns()

    def _calculate_returns(self) -> pd.DataFrame:
        """Calculate asset returns from price data."""
        returns = self.price_data.pct_change().dropna()
        if returns.empty:
            raise ValueError("Unable to calculate returns from price data")
        return returns

    def _validate_constraints(self, constraints: Dict) -> Dict:
        """Validate and set default constraints if needed."""
        if not isinstance(constraints, dict):
            raise TypeError("constraints must be a dictionary")
            
        defaults = {
            'max_position': 0.1,
            'max_sector': 0.3,
            'min_position': 0.01
        }
        return {**defaults, **constraints}

    def optimize(self, regime: str, risk_appetite: str, constraints: Optional[Dict] = None, valid_symbols: Optional[List[str]] = None) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Optimize portfolio based on market regime and risk appetite.
        
        Args:
            regime: Market regime ('Bullish', 'Bearish', or 'Neutral')
            risk_appetite: Risk appetite ('Conservative', 'Moderate', or 'Aggressive')
            constraints: Dictionary of portfolio constraints
            valid_symbols: List of valid asset symbols to include in optimization
            
        Returns:
            Tuple containing (weights, metrics) where:
            - weights: Dictionary mapping asset symbols to their portfolio weights
            - metrics: Dictionary of portfolio performance metrics
        """
        # Initialize default return values in case of exception
        if valid_symbols is None:
            valid_symbols = list(self.price_data.columns)
            
        # Default fallback portfolio (equal weight)
        total_assets = len(valid_symbols)
        fallback_weights = {symbol: 1.0/total_assets for symbol in valid_symbols}
        
        # Calculate basic metrics for fallback portfolio with robust error handling
        try:
            # Ensure valid_symbols is a list and contains valid columns
            valid_cols = [col for col in valid_symbols if col in self.returns.columns]
            
            if not valid_cols:
                # No valid columns, use safe defaults
                fallback_metrics = {
                    'expected_return': 0.0,
                    'volatility': 0.1,  # Default 10% volatility
                    'sharpe_ratio': 0.0,
                    'num_assets': total_assets,
                    'total_weight': 1.0,
                    'warning': 'No valid assets for return calculation'
                }
            else:
                # Calculate returns safely
                try:
                    # Properly calculate annualized returns as a Series
                    returns_data = self.returns[valid_cols].mean() * 252  # Annualized returns
                    
                    # Ensure returns is a pandas Series
                    if not isinstance(returns_data, pd.Series):
                        if isinstance(returns_data, (float, int, np.number)):
                            # Single value case
                            returns_data = pd.Series([returns_data] * len(valid_cols), index=valid_cols)
                        elif isinstance(returns_data, np.ndarray):
                            # Array case
                            returns_data = pd.Series(returns_data, index=valid_cols)
                        else:
                            # Fallback for unexpected types
                            logging.warning(f"Unexpected returns_data type: {type(returns_data)}")
                            returns_data = pd.Series([0.05] * len(valid_cols), index=valid_cols)  # 5% default return
                except Exception as e:
                    logging.warning(f"Error calculating returns: {str(e)}. Using default values.")
                    returns_data = pd.Series([0.05] * len(valid_cols), index=valid_cols)  # 5% default return
                
                # Calculate covariance matrix safely
                try:
                    cov_matrix = self._estimate_covariance(self.price_data[valid_cols])
                    volatility = np.sqrt(np.diagonal(cov_matrix))
                    
                    # Ensure volatility is a numpy array or pandas Series
                    if not isinstance(volatility, (np.ndarray, pd.Series)):
                        volatility = np.array([volatility] * len(valid_cols))
                        
                    # Replace any invalid values
                    volatility = np.nan_to_num(volatility, nan=0.15, posinf=0.15, neginf=0.15)
                except Exception as e:
                    logging.warning(f"Covariance estimation failed: {str(e)}. Using default volatility.")
                    volatility = np.array([0.15] * len(valid_cols))  # Default 15% volatility
                
                # Calculate Sharpe ratio safely
                try:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        # Ensure returns_data and volatility have the same length
                        if isinstance(returns_data, pd.Series) and len(returns_data) != len(volatility):
                            logging.warning(f"Length mismatch: returns_data ({len(returns_data)}) vs volatility ({len(volatility)})")
                            # Adjust lengths to match
                            min_len = min(len(returns_data), len(volatility))
                            returns_data = returns_data.iloc[:min_len] if len(returns_data) > min_len else returns_data
                            volatility = volatility[:min_len] if len(volatility) > min_len else volatility
                            
                        sharpe = np.divide(returns_data, volatility)
                        sharpe = np.nan_to_num(sharpe, nan=0.0, posinf=0.0, neginf=0.0)
                except Exception as e:
                    logging.warning(f"Sharpe ratio calculation failed: {str(e)}. Using default values.")
                    sharpe = np.array([0.0] * len(valid_cols))
                
                # Convert to float values for serialization with robust error handling
                try:
                    expected_return = float(np.nanmean(returns_data))
                    avg_volatility = float(np.nanmean(volatility))
                    avg_sharpe = float(np.nanmean(sharpe))
                except Exception as e:
                    logging.warning(f"Error converting metrics to float: {str(e)}. Using defaults.")
                    expected_return = 0.05  # 5% default return
                    avg_volatility = 0.15   # 15% default volatility
                    avg_sharpe = expected_return / avg_volatility if avg_volatility > 0 else 0.0
                
                fallback_metrics = {
                    'expected_return': expected_return,
                    'volatility': avg_volatility,
                    'sharpe_ratio': avg_sharpe,
                    'num_assets': total_assets,
                    'total_weight': 1.0,
                    'warning': 'Using equal weight fallback portfolio'
                }
        except Exception as e:
            logging.error(f"Error calculating fallback metrics: {str(e)}")
            # Use safe default values
            fallback_metrics = {
                'expected_return': 0.0,
                'volatility': 0.1,
                'sharpe_ratio': 0.0,
                'num_assets': total_assets,
                'total_weight': 1.0,
                'warning': f'Metrics calculation failed: {str(e)}'
            }

        # Validate regime parameter
        if regime not in ['Bullish', 'Bearish', 'Neutral']:
            logging.warning(f"Invalid regime '{regime}'. Using 'Neutral' as default.")
            regime = 'Neutral'
            
        # Validate risk_appetite parameter
        if risk_appetite not in ['Conservative', 'Moderate', 'Aggressive']:
            logging.warning(f"Invalid risk_appetite '{risk_appetite}'. Using 'Moderate' as default.")
            risk_appetite = 'Moderate'
        
        # Ensure constraints is a dictionary
        if constraints is None:
            constraints = {}
        
        try:
            constraints = self._validate_constraints(constraints)

            # Validate symbols
            invalid_symbols = set(valid_symbols) - set(self.price_data.columns)
            if invalid_symbols:
                raise ValueError(f"Invalid symbols provided: {invalid_symbols}")
                
            # Ensure we have at least one valid symbol
            if not valid_symbols or len(valid_symbols) == 0:
                logging.warning("No valid symbols provided for optimization")
                return fallback_weights, fallback_metrics

            filtered_price_data = self.price_data[valid_symbols]
            filtered_returns = self.returns[valid_symbols]

            # Ensure sector constraints are valid
            constraints['max_sector'] = min(1.0, constraints.get('max_sector', 0.3))

            # Covariance estimation with robust validation
            reg_cov_matrix = self._estimate_covariance(filtered_price_data)

            # Map risk appetite to optimization parameters
            risk_targets = {
                'Conservative': {'target_volatility': 0.10, 'risk_aversion': 2.0},
                'Moderate': {'target_volatility': 0.15, 'risk_aversion': 1.0},
                'Aggressive': {'target_volatility': 0.20, 'risk_aversion': 0.5}
            }
            
            risk_params = risk_targets.get(risk_appetite, risk_targets['Moderate'])
            
            # Optimize portfolio based on regime and risk appetite with improved error handling
            try:
                # Set up and configure efficient frontier
                ef = self._configure_efficient_frontier(
                    filtered_returns,
                    reg_cov_matrix,
                    constraints,
                    valid_symbols
                )
                
                # Initialize weights to None to detect optimization failures
                weights = None
                
                # Optimize based on regime
                if regime == 'Bearish':
                    try:
                        # For bearish regime, minimize volatility
                        weights = ef.min_volatility()
                    except Exception as vol_error:
                        logging.warning(f"Min volatility optimization failed: {str(vol_error)}. Trying with more relaxed constraints.")
                        try:
                            # Try with more relaxed constraints
                            relaxed_constraints = constraints.copy()
                            relaxed_constraints['min_position'] = 0.0  # Allow zero positions
                            ef = self._configure_efficient_frontier(filtered_returns, reg_cov_matrix, relaxed_constraints, valid_symbols)
                            weights = ef.min_volatility()
                        except Exception as e:
                            logging.error(f"Relaxed min volatility also failed: {str(e)}. Will use fallback weights.")
                            # Immediately return fallback weights instead of continuing
                            return fallback_weights, {**fallback_metrics, 'warning': f'Bearish optimization failed: {str(e)}. Using equal weight portfolio.'}
                        
                elif regime == 'Bullish':
                    try:
                        # For bullish regime, maximize Sharpe ratio
                        # Set solver properties directly on the object before calling max_sharpe
                        ef.solver = 'ECOS'
                        ef.solver_options = {
                            'max_iters': 1000,
                            'abstol': 1e-8,
                            'reltol': 1e-8
                        }
                        weights = ef.max_sharpe(risk_free_rate=0.02)
                    except Exception as sharpe_error:
                        logging.warning(f"Max Sharpe optimization failed: {str(sharpe_error)}. Trying alternative approach.")
                        try:
                            # First try: Maximize Sharpe with relaxed constraints
                            relaxed_constraints = constraints.copy()
                            relaxed_constraints['min_position'] = 0.0
                            relaxed_constraints['max_position'] = min(1.0, constraints['max_position'] * 1.2)
                            ef = self._configure_efficient_frontier(filtered_returns, reg_cov_matrix, relaxed_constraints, valid_symbols)
                            # Set solver properties directly on the object
                            ef.solver = 'ECOS'
                            ef.solver_options = {
                                'max_iters': 1000,
                                'abstol': 1e-8,
                                'reltol': 1e-8
                            }
                            weights = ef.max_sharpe(risk_free_rate=0.02)
                        except Exception as e1:
                            logging.warning(f"Relaxed max_sharpe failed: {str(e1)}. Trying efficient return.")
                            try:
                                # Second try: Use efficient_return with improved solver settings
                                ef = self._configure_efficient_frontier(filtered_returns, reg_cov_matrix, constraints, valid_symbols)
                                ef.solver = 'SCS'
                                ef.solver_options = {
                                    'max_iters': 2500,
                                    'eps': 1e-5,
                                    'normalize': True
                                }
                                
                                # Target a reasonable return with more conservative bounds
                                target_return = filtered_returns.mean().mean() * 252 * 1.1  # 10% higher than average
                                target_return = max(0.03, min(0.20, target_return))  # More conservative bounds
                                weights = ef.efficient_return(target_return=target_return, market_neutral=False)
                            except Exception as e2:
                                logging.warning(f"Efficient return failed: {str(e2)}. Using min volatility.")
                                try:
                                    ef = self._configure_efficient_frontier(filtered_returns, reg_cov_matrix, constraints, valid_symbols)
                                    ef.solver = 'OSQP'
                                    ef.solver_options = {
                                        'max_iter': 5000,
                                        'eps_abs': 1e-8,
                                        'eps_rel': 1e-8
                                    }
                                    weights = ef.min_volatility()
                                except Exception as e3:
                                    logging.error(f"All optimization methods failed for Bullish regime: {str(e3)}. Will use fallback weights.")
                                    # Immediately return fallback weights instead of continuing
                                    return fallback_weights, {**fallback_metrics, 'warning': f'Bullish optimization failed: {str(e3)}. Using equal weight portfolio.'}
                                
                elif regime == 'Neutral':
                    # For neutral regime, target specific risk level
                    try:
                        ef.solver = 'ECOS'
                        ef.solver_options = {
                            'max_iters': 1000,
                            'abstol': 1e-8,
                            'reltol': 1e-8,
                            'feastol': 1e-8
                        }
                        weights = ef.efficient_risk(
                            target_volatility=risk_params['target_volatility'],
                            risk_free_rate=0.02
                        )
                    except Exception as risk_error:
                        logging.warning(f"Efficient risk optimization failed: {str(risk_error)}. Trying alternative approaches.")
                        try:
                            # First fallback: Try with relaxed constraints and SCS solver
                            relaxed_constraints = constraints.copy()
                            relaxed_constraints['min_position'] = 0.0  # Allow zero positions
                            relaxed_constraints['max_position'] = min(1.0, constraints['max_position'] + 0.1)  # Increase max position
                            ef = self._configure_efficient_frontier(filtered_returns, reg_cov_matrix, relaxed_constraints, valid_symbols)
                            ef.solver = 'SCS'
                            ef.solver_options = {
                                'max_iters': 2500,
                                'eps': 1e-5,
                                'normalize': True,
                                'acceleration_lookback': 20
                            }
                            weights = ef.efficient_risk(
                                target_volatility=risk_params['target_volatility'] * 1.1,  # Allow 10% higher volatility
                                risk_free_rate=0.02
                            )
                        except Exception as e1:
                            logging.warning(f"Relaxed efficient_risk failed: {str(e1)}. Trying min_volatility.")
                            try:
                                # Second fallback: Try min_volatility with OSQP solver
                                ef = self._configure_efficient_frontier(filtered_returns, reg_cov_matrix, constraints, valid_symbols)
                                ef.solver = 'OSQP'
                                ef.solver_options = {
                                    'max_iter': 5000,
                                    'eps_abs': 1e-8,
                                    'eps_rel': 1e-8,
                                    'polish': True,
                                    'adaptive_rho': True
                                }
                                weights = ef.min_volatility()
                            except Exception as e2:
                                logging.warning(f"Min volatility also failed: {str(e2)}. Using HRP as final fallback.")
                                # Final fallback: Use hierarchical risk parity (HRP) which is more robust
                                try:
                                    from pypfopt import hierarchical_portfolio as hp
                                    hrp = hp.HRPOpt(filtered_returns, cov_matrix=reg_cov_matrix)
                                    weights = hrp.optimize()
                                except Exception as e3:
                                    logging.error(f"All optimization methods failed: {str(e3)}. Will use fallback weights.")
                                    # Immediately return fallback weights instead of continuing
                                    return fallback_weights, {**fallback_metrics, 'warning': f'Neutral optimization failed: {str(e3)}. Using equal weight portfolio.'}
                else:
                    # This should never happen due to validation above, but just in case
                    logging.warning(f"Unhandled regime: {regime}. Using Neutral approach.")
                    ef.solver = 'ECOS'
                    try:
                        weights = ef.efficient_risk(
                            target_volatility=risk_params['target_volatility'],
                            risk_free_rate=0.02
                        )
                    except Exception as e:
                        logging.warning(f"Efficient risk failed for unhandled regime: {str(e)}. Using min volatility.")
                        try:
                            weights = ef.min_volatility()
                        except Exception as e2:
                            logging.error(f"Min volatility also failed for unhandled regime: {str(e2)}. Will use fallback weights.")
                            # Immediately return fallback weights instead of continuing
                            return fallback_weights, {**fallback_metrics, 'warning': f'Unhandled regime optimization failed: {str(e2)}. Using equal weight portfolio.'}
                
                # Check if optimization failed completely and use fallback weights if needed
                if weights is None:
                    logging.warning("All optimization methods failed. Using equal weight fallback.")
                    return fallback_weights, {**fallback_metrics, 'warning': 'All optimization methods failed. Using equal weight portfolio.'}
                
                # Verify weights is a dictionary before proceeding
                if not isinstance(weights, dict):
                    try:
                        # Try to convert to dictionary if it's another format
                        weights = dict(weights)
                    except (TypeError, ValueError) as e:
                        logging.error(f"Weights is not a dictionary and cannot be converted: {str(e)}")
                        return fallback_weights, {**fallback_metrics, 'warning': 'Optimization produced invalid weights format. Using equal weight portfolio.'}
                
                # Clean weights with improved precision and error handling
                try:
                    weights = ef.clean_weights(cutoff=0.0001)
                    
                    # Verify weights are not empty
                    if not weights:
                        logging.warning("clean_weights returned empty weights. Using fallback weights.")
                        return fallback_weights, {**fallback_metrics, 'warning': 'Optimization produced empty weights. Using equal weight portfolio.'}
                    
                    # Ensure weights sum to 1.0 (sometimes clean_weights can result in sum < 1)
                    weight_sum = sum(weights.values())
                    if abs(weight_sum - 1.0) > 1e-5:  # If weights don't sum to approximately 1
                        if weight_sum > 0:  # Only normalize if sum is positive
                            logging.info(f"Adjusting weights to sum to 1.0 (current sum: {weight_sum})")
                            # Normalize weights to sum to 1
                            weights = {k: v/weight_sum for k, v in weights.items()}
                        else:
                            logging.warning("Weights sum to zero or negative. Using fallback weights.")
                            return fallback_weights, {**fallback_metrics, 'warning': 'Optimization produced invalid weights. Using equal weight portfolio.'}
                except Exception as clean_error:
                    logging.error(f"Error cleaning weights: {str(clean_error)}. Using fallback weights.")
                    return fallback_weights, {**fallback_metrics, 'warning': f'Error cleaning weights: {str(clean_error)}. Using equal weight portfolio.'}
                
                # Get performance metrics with robust error handling
                try:
                    perf = ef.portfolio_performance(risk_free_rate=0.02)
                except Exception as perf_error:
                    logging.warning(f"Error calculating portfolio performance: {str(perf_error)}")
                    # Calculate performance manually with additional validation
                    try:
                        expected_return = sum(weights[asset] * filtered_returns[asset].mean() * 252 for asset in weights if asset in filtered_returns)
                        
                        # More robust portfolio volatility calculation
                        portfolio_vol = 0.0
                        try:
                            # Get indices for valid symbols
                            valid_indices = {symbol: i for i, symbol in enumerate(valid_symbols) if symbol in weights}
                            
                            # Calculate portfolio variance
                            portfolio_var = 0.0
                            for i in weights:
                                if i not in valid_indices:
                                    continue
                                for j in weights:
                                    if j not in valid_indices:
                                        continue
                                    i_idx = valid_indices[i]
                                    j_idx = valid_indices[j]
                                    if i_idx < reg_cov_matrix.shape[0] and j_idx < reg_cov_matrix.shape[1]:
                                        portfolio_var += weights[i] * weights[j] * reg_cov_matrix[i_idx, j_idx]
                            
                            portfolio_vol = np.sqrt(max(0, portfolio_var))  # Ensure non-negative
                        except Exception as vol_error:
                            logging.warning(f"Error calculating portfolio volatility: {str(vol_error)}. Using default value.")
                            portfolio_vol = 0.15  # Default 15% volatility
                          
                        # Calculate Sharpe ratio with validation
                        sharpe = (expected_return - 0.02) / portfolio_vol if portfolio_vol > 0 else 0
                        perf = (expected_return, portfolio_vol, sharpe)
                    except Exception as manual_error:
                        logging.error(f"Manual performance calculation failed: {str(manual_error)}. Using default values.")
                        perf = (0.05, 0.15, 0.2)  # Default values
                  
                # Ensure all metrics are valid numbers
                try:
                    metrics = {
                        'expected_return': float(perf[0]),
                        'volatility': float(perf[1]),
                        'sharpe_ratio': float(perf[2]),
                        'num_assets': len([w for w in weights.values() if w > constraints.get('min_position', 0.01)]),
                        'total_weight': sum(weights.values())
                    }
                      
                    # Validate metrics
                    for key in ['expected_return', 'volatility', 'sharpe_ratio']:
                        if not np.isfinite(metrics[key]):
                            logging.warning(f"Invalid {key}: {metrics[key]}. Using default value.")
                            if key == 'expected_return':
                                metrics[key] = 0.05
                            elif key == 'volatility':
                                metrics[key] = 0.15
                            else:  # sharpe_ratio
                                metrics[key] = metrics['expected_return'] / metrics['volatility'] if metrics['volatility'] > 0 else 0
                except Exception as metrics_error:
                    logging.error(f"Error creating metrics dictionary: {str(metrics_error)}. Using default metrics.")
                    metrics = {
                        'expected_return': 0.05,
                        'volatility': 0.15,
                        'sharpe_ratio': 0.2,
                        'num_assets': len(weights),
                        'total_weight': sum(weights.values()),
                        'warning': 'Error calculating portfolio metrics. Using default values.'
                    }

                return weights, metrics
            except Exception as opt_error:
                logging.error(f"All optimization methods failed: {str(opt_error)}. Using equal weight fallback.")
                # Use equal weight fallback with more descriptive warning that includes the actual error
                return fallback_weights, {**fallback_metrics, 'warning': f'Optimization failed: {str(opt_error)}. Using equal weight portfolio.'}

        except Exception as e:
            logging.error(f"Portfolio optimization failed: {str(e)}")
            fallback_metrics['warning'] = f'Using equal weight fallback due to: {str(e)}'
            return fallback_weights, fallback_metrics

    def _estimate_covariance(self, price_data: pd.DataFrame) -> np.ndarray:
        """Estimate the covariance matrix using Ledoit-Wolf shrinkage.

        Args:
            price_data: DataFrame of asset prices

        Returns:
            Estimated covariance matrix with stability adjustments
        """
        try:
            # Validate input data first
            if price_data.empty:
                raise ValueError("Empty price data provided")
                
            if price_data.shape[1] < 2:
                # Special case: only one asset
                try:
                    variance = price_data.pct_change().var().iloc[0] * 252
                    if np.isnan(variance) or variance <= 0:
                        variance = 0.04  # Default 20% volatility squared
                    return np.array([[variance]])
                except Exception as e:
                    logging.warning(f"Error calculating variance for single asset: {str(e)}")
                    return np.array([[0.04]])  # Default 20% volatility squared
                
            # Check for sufficient data points
            min_history = 60  # Require at least 60 data points for reliable estimation
            if price_data.shape[0] < min_history:
                logging.warning(f"Insufficient price history: {price_data.shape[0]} < {min_history} days")
                # Fall back to sample covariance with stability factor
                try:
                    sample_cov = risk_models.sample_cov(price_data, frequency=252)
                    # Ensure the matrix is valid
                    if not np.all(np.isfinite(sample_cov)):
                        raise ValueError("Sample covariance contains non-finite values")
                    return sample_cov + np.eye(sample_cov.shape[0]) * 1e-4  # Increased stability factor
                except Exception as e:
                    logging.warning(f"Sample covariance failed for insufficient history: {str(e)}")
                    # Create a simple diagonal covariance matrix
                    n_assets = price_data.shape[1]
                    vols = price_data.pct_change().std().fillna(0.02) * np.sqrt(252)
                    vols = np.clip(vols, 0.1, 0.4)  # Reasonable volatility bounds
                    return np.diag(vols**2)
            
            # Use Ledoit-Wolf shrinkage for base estimation with better error handling
            try:
                reg_cov_matrix = risk_models.CovarianceShrinkage(
                    price_data,
                    frequency=252
                ).ledoit_wolf()
            except Exception as e:
                logging.warning(f"Ledoit-Wolf shrinkage failed: {str(e)}. Trying sample covariance.")
                try:
                    reg_cov_matrix = risk_models.sample_cov(price_data, frequency=252)
                except Exception as inner_e:
                    logging.warning(f"Sample covariance also failed: {str(inner_e)}. Using robust estimator.")
                    # Use a more robust estimator
                    returns = price_data.pct_change().dropna()
                    # Calculate pairwise correlations and volatilities separately
                    vols = returns.std() * np.sqrt(252)
                    vols = vols.clip(0.1, 0.4)  # Reasonable bounds
                    corr = returns.corr().fillna(0)
                    # Ensure correlation matrix is valid
                    corr = np.clip(corr, -0.95, 0.95)
                    np.fill_diagonal(corr.values, 1.0)
                    # Reconstruct covariance matrix
                    vol_matrix = np.diag(vols)
                    reg_cov_matrix = vol_matrix @ corr @ vol_matrix
                    return reg_cov_matrix
            
            # Validate matrix properties
            if not np.all(np.isfinite(reg_cov_matrix)):
                logging.warning("Covariance matrix contains non-finite values, replacing with robust estimate")
                # Create a robust diagonal matrix
                n_assets = price_data.shape[1]
                vols = price_data.pct_change().std().fillna(0.02) * np.sqrt(252)
                vols = np.clip(vols, 0.1, 0.4)  # Reasonable volatility bounds
                return np.diag(vols**2)
                
            # Check positive definiteness with improved error handling
            try:
                eigenvals = np.linalg.eigvals(reg_cov_matrix)
                min_eigenval = np.min(eigenvals)
                if min_eigenval <= 0:
                    logging.warning(f"Covariance matrix is not positive definite, min eigenvalue: {min_eigenval}")
                    # Add stronger stability term based on the magnitude of the issue
                    stability_factor = max(1e-4, abs(min_eigenval) * 2)
                    reg_cov_matrix += np.eye(reg_cov_matrix.shape[0]) * stability_factor
                    logging.info(f"Applied stability factor of {stability_factor} to covariance matrix")
            except np.linalg.LinAlgError:
                logging.warning("Eigenvalue computation failed, applying stronger correction")
                stability_factor = 1e-3  # Increased from 1e-4
                reg_cov_matrix += np.eye(reg_cov_matrix.shape[0]) * stability_factor
                
            # Always add minimal stability term to ensure numerical stability
            stability_factor = 1e-5  # Increased from 1e-6
            reg_cov_matrix += np.eye(reg_cov_matrix.shape[0]) * stability_factor
            
        except Exception as e:
            logging.error(f"All covariance estimation methods failed: {str(e)}")
            # Last resort: create a diagonal matrix with conservative volatility estimates
            try:
                n_assets = price_data.shape[1]
                # Use a more conservative approach for the fallback
                vols = np.ones(n_assets) * 0.2  # Default 20% volatility for all assets
                
                # Try to get some asset-specific information if possible
                try:
                    asset_vols = price_data.pct_change().std().fillna(0.2) * np.sqrt(252)
                    asset_vols = np.clip(asset_vols, 0.15, 0.4)  # More conservative bounds
                    if len(asset_vols) == n_assets:
                        vols = asset_vols
                except:
                    pass  # Stick with the default if this fails
                    
                # Create diagonal covariance matrix
                reg_cov_matrix = np.diag(vols**2)
                
                # Add small off-diagonal elements to represent some correlation
                corr_factor = 0.2  # Modest correlation
                for i in range(n_assets):
                    for j in range(n_assets):
                        if i != j:
                            reg_cov_matrix[i, j] = corr_factor * vols[i] * vols[j]
                            
                logging.warning("Using fallback diagonal covariance matrix with modest correlations")
            except Exception as final_e:
                logging.error(f"Even fallback covariance creation failed: {str(final_e)}")
                # Absolute last resort - identity matrix scaled by reasonable variance
                n_assets = max(1, price_data.shape[1])
                reg_cov_matrix = np.eye(n_assets) * 0.04  # 20% volatility squared
            
        return reg_cov_matrix

    def _configure_efficient_frontier(
        self,
        returns: pd.DataFrame,
        cov_matrix: np.ndarray,
        constraints: Dict,
        valid_symbols: List[str]
    ) -> EfficientFrontier:
        """Configure the efficient frontier with objectives and constraints.

        Args:
            returns: Asset returns
            cov_matrix: Covariance matrix
            constraints: Portfolio constraints
            valid_symbols: List of valid asset symbols

        Returns:
            Configured EfficientFrontier object
        """
        try:
            # Ensure returns is a pandas Series with proper index
            if not isinstance(returns, pd.Series):
                # If returns is a DataFrame, convert to Series of mean returns
                if isinstance(returns, pd.DataFrame):
                    returns = returns.mean() * 252  # Annualize returns
                # If returns is a numpy array or other type, convert to Series
                elif isinstance(returns, (np.ndarray, list)):
                    returns = pd.Series(returns, index=valid_symbols)
                else:
                    logging.warning(f"Unexpected returns type: {type(returns)}. Creating default returns.")
                    returns = pd.Series([0.05] * len(valid_symbols), index=valid_symbols)  # Default 5% return
            
            # Validate returns has the right shape
            if len(returns) != len(valid_symbols):
                logging.warning(f"Returns length mismatch: {len(returns)} vs {len(valid_symbols)}. Creating default returns.")
                returns = pd.Series([0.05] * len(valid_symbols), index=valid_symbols)  # Default 5% return
            
            # Ensure no NaN values in returns
            if returns.isna().any():
                logging.warning("NaN values found in returns. Replacing with mean values.")
                mean_return = returns.mean()
                if np.isnan(mean_return):
                    mean_return = 0.05  # Default if all are NaN
                returns = returns.fillna(mean_return)
            
            # Validate returns for extreme values
            returns = returns.clip(-0.5, 0.5)  # Limit extreme returns
                
            # Ensure covariance matrix is positive definite with stronger correction
            try:
                # Check eigenvalues
                min_eigenval = np.min(np.linalg.eigvals(cov_matrix))
                if min_eigenval <= 0:
                    logging.warning(f"Covariance matrix not positive definite. Min eigenvalue: {min_eigenval}")
                    # Apply stronger correction
                    stability_factor = max(1e-4, abs(min_eigenval) * 2)
                    cov_matrix += np.eye(cov_matrix.shape[0]) * stability_factor
                    logging.info(f"Applied stability factor of {stability_factor} to covariance matrix")
            except np.linalg.LinAlgError:
                logging.warning("Eigenvalue computation failed, applying stronger correction")
                stability_factor = 1e-3
                cov_matrix += np.eye(cov_matrix.shape[0]) * stability_factor
                
            # Create EfficientFrontier instance with validated inputs and relaxed bounds
            # Slightly relax the min position constraint to help solver convergence
            min_position = max(0.0, constraints['min_position'] - 0.001)  # Slightly relax lower bound
            max_position = min(1.0, constraints['max_position'] + 0.01)  # Slightly relax upper bound
            
            # Create the EfficientFrontier object
            try:
                ef = EfficientFrontier(
                    expected_returns=returns,
                    cov_matrix=cov_matrix,
                    weight_bounds=(min_position, max_position),
                    verbose=False
                )
                
                # Set default solver and solver options
                ef.solver = 'ECOS'
                ef.solver_options = {
                    'max_iters': 1000,  # Increase max iterations
                    'abstol': 1e-8,    # Absolute tolerance
                    'reltol': 1e-8     # Relative tolerance
                }

                # Add L2 regularization with reduced gamma to avoid over-constraining
                ef.add_objective(objective_functions.L2_reg, gamma=0.1)

                # Add sector constraints if defined, with relaxed bounds
                if self.sectors:
                    try:
                        sector_mapper = {asset: self.sectors.get(asset, 'Other') for asset in valid_symbols}
                        
                        # Slightly relax sector constraints
                        max_sector = min(1.0, constraints['max_sector'] + 0.05)  # Add 5% slack
                        
                        sector_bounds = {sector: (0.0, max_sector) 
                                      for sector in set(sector_mapper.values())}
                        
                        # Only add sector constraints if we have more than one sector
                        if len(set(sector_mapper.values())) > 1:
                            ef.add_sector_constraints(
                                sector_mapper, 
                                sector_lower={s: v[0] for s, v in sector_bounds.items()},
                                sector_upper={s: v[1] for s, v in sector_bounds.items()}
                            )
                    except Exception as sector_error:
                        logging.warning(f"Error adding sector constraints: {str(sector_error)}. Proceeding without sector constraints.")
                
                return ef
            except Exception as ef_error:
                logging.error(f"Error creating EfficientFrontier object: {str(ef_error)}")
                raise
        except Exception as e:
            logging.error(f"Error configuring efficient frontier: {str(e)}")
            raise