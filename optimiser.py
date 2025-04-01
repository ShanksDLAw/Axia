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

            # Set up and configure efficient frontier
            try:
                ef = self._configure_efficient_frontier(
                    filtered_returns,
                    reg_cov_matrix,
                    constraints,
                    valid_symbols
                )

                # Map risk appetite to optimization parameters
                risk_targets = {
                    'Conservative': {'target_volatility': 0.10, 'risk_aversion': 2.0},
                    'Moderate': {'target_volatility': 0.15, 'risk_aversion': 1.0},
                    'Aggressive': {'target_volatility': 0.20, 'risk_aversion': 0.5}
                }
                
                risk_params = risk_targets.get(risk_appetite, risk_targets['Moderate'])
                
                # Optimize portfolio based on regime and risk appetite with improved error handling
                try:
                    if regime == 'Bearish':
                        try:
                            weights = ef.min_volatility()
                        except Exception as vol_error:
                            logging.warning(f"Min volatility optimization failed: {str(vol_error)}. Trying with more relaxed constraints.")
                            # Try with more relaxed constraints
                            relaxed_constraints = constraints.copy()
                            relaxed_constraints['min_position'] = 0.0  # Allow zero positions
                            ef = self._configure_efficient_frontier(filtered_returns, reg_cov_matrix, relaxed_constraints, valid_symbols)
                            weights = ef.min_volatility()
                    elif regime == 'Bullish':
                        # Try max_sharpe with improved error handling and no solver parameter
                        try:
                            weights = ef.max_sharpe(risk_free_rate=0.02)
                        except Exception as sharpe_error:
                            logging.warning(f"Max Sharpe optimization failed: {str(sharpe_error)}. Trying alternative approach.")
                            try:
                                # First try: Maximize Sharpe with relaxed constraints
                                relaxed_constraints = constraints.copy()
                                relaxed_constraints['min_position'] = 0.0
                                relaxed_constraints['max_position'] = min(1.0, constraints['max_position'] * 1.2)
                                ef = self._configure_efficient_frontier(filtered_returns, reg_cov_matrix, relaxed_constraints, valid_symbols)
                                weights = ef.max_sharpe(risk_free_rate=0.02)
                            except Exception as e1:
                                logging.warning(f"Relaxed max_sharpe failed: {str(e1)}. Trying efficient return.")
                                try:
                                    # Second try: Use efficient_return with dynamic target
                                    target_return = max(0.05, filtered_returns.mean().mean() * 252)
                                    weights = ef.efficient_return(target_return=target_return)
                                except Exception as e2:
                                    logging.warning(f"Efficient return failed: {str(e2)}. Using min volatility.")
                                    weights = ef.min_volatility()
                    
                    # Clean weights with improved precision
                    weights = ef.clean_weights(cutoff=0.0001)
                    # Try different approaches in sequence with improved solver configuration
                    try:
                        # First try: Reset with more relaxed constraints and improved solver settings
                        relaxed_constraints = constraints.copy()
                        relaxed_constraints['min_position'] = max(0.0, constraints['min_position'] - 0.005)
                        relaxed_constraints['max_position'] = min(1.0, constraints['max_position'] + 0.05)
                        ef = self._configure_efficient_frontier(filtered_returns, reg_cov_matrix, relaxed_constraints, valid_symbols)
                        
                        # Configure solver for better numerical stability
                        ef.solver = 'ECOS'
                        ef.solver_options = {
                            'max_iters': 1000,
                            'abstol': 1e-8,
                            'reltol': 1e-8,
                            'feastol': 1e-8
                        }
                        
                        weights = ef.max_sharpe(risk_free_rate=0.02)
                    except Exception as e1:
                        logging.warning(f"Relaxed constraints approach failed: {str(e1)}. Trying efficient return.")
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
                            logging.warning(f"Efficient return approach failed: {str(e2)}. Using robust min volatility.")
                            # Last try: Fall back to min_volatility with most stable solver
                            ef = self._configure_efficient_frontier(filtered_returns, reg_cov_matrix, constraints, valid_symbols)
                            ef.solver = 'OSQP'
                            ef.solver_options = {
                                'max_iter': 5000,
                                'eps_abs': 1e-8,
                                'eps_rel': 1e-8
                            }
                            weights = ef.min_volatility()
                        
                        weights = ef.clean_weights(cutoff=0.001)  # Slightly higher cutoff for more stability
                    else:  # Neutral
                        # Try efficient_risk with multiple fallback options and improved solver settings
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
                                        logging.error(f"All optimization methods failed: {str(e3)}. Using equal weight fallback.")
                                        return fallback_weights, {**fallback_metrics, 'warning': 'All optimization methods failed'}
                            
                        # Clean weights with higher cutoff for stability
                        weights = ef.clean_weights(cutoff=0.001)

                            
                    # Apply a more lenient cutoff to avoid infeasible solutions
                    weights = ef.clean_weights(cutoff=max(0.0005, constraints['min_position'] * 0.25))
                    
                    # Ensure weights sum to 1.0 (sometimes clean_weights can result in sum < 1)
                    weight_sum = sum(weights.values())
                    if abs(weight_sum - 1.0) > 1e-5:  # If weights don't sum to approximately 1
                        logging.info(f"Adjusting weights to sum to 1.0 (current sum: {weight_sum})")
                        # Normalize weights to sum to 1
                        weights = {k: v/weight_sum for k, v in weights.items()}
                    
                    # Get performance metrics with robust error handling
                    try:
                        perf = ef.portfolio_performance(risk_free_rate=0.02)
                    except Exception as perf_error:
                        logging.warning(f"Error calculating portfolio performance: {str(perf_error)}")
                        # Calculate performance manually
                        expected_return = sum(weights[asset] * returns[asset] for asset in weights)
                        portfolio_vol = np.sqrt(
                            sum(weights[i] * weights[j] * cov_matrix[list(valid_symbols).index(i)][list(valid_symbols).index(j)]
                                for i in weights for j in weights)
                        )
                        sharpe = (expected_return - 0.02) / portfolio_vol if portfolio_vol > 0 else 0
                        perf = (expected_return, portfolio_vol, sharpe)
                    
                except Exception as opt_error:
                    logging.error(f"All optimization methods failed: {str(opt_error)}. Using equal weight fallback.")
                    # Use equal weight fallback
                    return fallback_weights, {**fallback_metrics, 'warning': f'Optimization failed: {str(opt_error)}'}
                
                metrics = {
                    'expected_return': float(perf[0]),
                    'volatility': float(perf[1]),
                    'sharpe_ratio': float(perf[2]),
                    'num_assets': len([w for w in weights.values() if w > constraints['min_position']]),
                    'total_weight': sum(weights.values())
                }

                return weights, metrics
            except Exception as e:
                logging.error(f"Efficient frontier optimization failed: {str(e)}")
                fallback_metrics['warning'] = f'Using equal weight fallback due to: {str(e)}'
                return fallback_weights, fallback_metrics

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
                variance = price_data.pct_change().var().iloc[0] * 252
                return np.array([[variance]])
                
            # Check for sufficient data points
            min_history = 60  # Require at least 60 data points for reliable estimation
            if price_data.shape[0] < min_history:
                logging.warning(f"Insufficient price history: {price_data.shape[0]} < {min_history} days")
                # Fall back to sample covariance with stability factor
                sample_cov = risk_models.sample_cov(price_data, frequency=252)
                return sample_cov + np.eye(sample_cov.shape[0]) * 1e-5
            
            # Use Ledoit-Wolf shrinkage for base estimation
            reg_cov_matrix = risk_models.CovarianceShrinkage(
                price_data,
                frequency=252
            ).ledoit_wolf()
            
            # Validate matrix properties
            if not np.all(np.isfinite(reg_cov_matrix)):
                raise ValueError("Covariance matrix contains non-finite values")
                
            # Check positive definiteness
            try:
                if not np.all(np.linalg.eigvals(reg_cov_matrix) > 0):
                    logging.warning("Covariance matrix is not positive definite, applying correction")
                    # Add minimal stability term
                    stability_factor = 1e-5
                    reg_cov_matrix += np.eye(reg_cov_matrix.shape[0]) * stability_factor
            except np.linalg.LinAlgError:
                logging.warning("Eigenvalue computation failed, applying stronger correction")
                stability_factor = 1e-4
                reg_cov_matrix += np.eye(reg_cov_matrix.shape[0]) * stability_factor
                
            # Add minimal stability term
            stability_factor = 1e-6
            reg_cov_matrix += np.eye(reg_cov_matrix.shape[0]) * stability_factor
            
        except Exception as e:
            logging.warning(f"Ledoit-Wolf estimation failed: {str(e)}. Using sample covariance.")
            try:
                reg_cov_matrix = risk_models.sample_cov(price_data, frequency=252)
                reg_cov_matrix += np.eye(reg_cov_matrix.shape[0]) * 1e-5
            except Exception as inner_e:
                logging.error(f"Sample covariance estimation also failed: {str(inner_e)}")
                # Last resort: create a diagonal matrix with reasonable volatility estimates
                n_assets = price_data.shape[1]
                vols = price_data.pct_change().std().fillna(0.01) * np.sqrt(252)
                vols = np.clip(vols, 0.05, 0.5)  # Reasonable volatility bounds
                reg_cov_matrix = np.diag(vols**2)
            
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
            
            # Set solver options for better convergence
            solver_options = {
                'max_iters': 1000,  # Increase max iterations
                'tol': 1e-6        # Slightly relax tolerance
            }
            
            ef = EfficientFrontier(
                expected_returns=returns,
                cov_matrix=cov_matrix,
                weight_bounds=(min_position, max_position),
                verbose=False
            )

            # Add L2 regularization with reduced gamma to avoid over-constraining
            ef.add_objective(objective_functions.L2_reg, gamma=0.1)

            # Add sector constraints if defined, with relaxed bounds
            if self.sectors:
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
                
            return ef
        except Exception as e:
            logging.error(f"Error configuring efficient frontier: {str(e)}")
            raise