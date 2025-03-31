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
        fallback_metrics = {
            'expected_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'num_assets': total_assets,
            'total_weight': 1.0,
            'warning': ''  # Add warning field to fallback metrics
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

                # Optimize portfolio based on regime and risk appetite
                if regime == 'Bearish':
                    weights = ef.min_volatility()
                elif regime == 'Bullish':
                    weights = ef.max_sharpe()
                else:  # Neutral
                    weights = ef.efficient_risk(target_volatility=0.15)

                weights = ef.clean_weights(cutoff=constraints['min_position'])
                perf = ef.portfolio_performance()
                
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
            ef = EfficientFrontier(
                returns,
                cov_matrix,
                weight_bounds=(constraints['min_position'], constraints['max_position'])
            )

            # Add L2 regularization to reduce extreme weights
            ef.add_objective(objective_functions.L2_reg, gamma=0.5)

            # Add sector constraints if defined
            if self.sectors:
                sector_mapper = {asset: self.sectors.get(asset, 'Other') for asset in valid_symbols}
                sector_bounds = {sector: (0.0, constraints['max_sector']) 
                               for sector in set(sector_mapper.values())}
                ef.add_sector_constraints(
                    sector_mapper, 
                    sector_lower={s: v[0] for s, v in sector_bounds.items()},
                    sector_upper={s: v[1] for s, v in sector_bounds.items()}
                )
                
            return ef
        except Exception as e:
            logging.error(f"Error configuring efficient frontier: {str(e)}")
            raise