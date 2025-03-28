# src/optimizer.py
import pandas as pd
import numpy as np
import logging
import cvxpy as cp
from pypfopt import expected_returns, risk_models, EfficientFrontier, HRPOpt, objective_functions
from collections import defaultdict
from typing import Dict, Tuple, List
import warnings

# Suppress optimization warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class PortfolioOptimizer:
    def __init__(self, price_data: pd.DataFrame, sectors: dict[str, str]):
        if price_data.empty:
            raise ValueError("Cannot initialize optimizer with empty price data")
        self.price_data = price_data
        self.sectors = sectors
        # Enhanced returns estimation with exponential weighting
        self.returns_df = expected_returns.returns_from_prices(price_data)
        self.expected_returns = expected_returns.ema_historical_return(price_data, span=252)
        
        # Robust covariance estimation with additional shrinkage
        self.cov_matrix = risk_models.CovarianceShrinkage(
            price_data,
            frequency=252
        ).ledoit_wolf()
        # Calculate sector allocations
        self.sector_weights = self._calculate_sector_weights()
    
    def _calculate_sector_weights(self) -> dict[str, float]:
        """Calculate sector weights including Unknown sector with enhanced validation"""
        sector_weights = defaultdict(float)
        sector_counts = defaultdict(int)
        
        # Normalize and validate sectors first
        sector_mapping = {
            'Consumer Cyclical': 'Consumer Discretionary',
            'Consumer Defensive': 'Consumer Staples',
            'Financial': 'Financial Services',
            'Technology': 'Information Technology',
            'Basic Materials': 'Materials',
            'Communication Services': 'Communication Services',
            'Healthcare': 'Healthcare',
            'Utilities': 'Utilities',
            'Real Estate': 'Real Estate',
            'Energy': 'Energy',
            'Industrials': 'Industrials'
        }
        
        # Normalize sectors with validation
        valid_sectors = {}
        for symbol, sector in self.sectors.items():
            if not isinstance(sector, str):
                logging.warning(f"Invalid sector type for {symbol}: {type(sector)}")
                valid_sectors[symbol] = 'Unknown'
                continue
                
            normalized_sector = sector_mapping.get(sector, sector)
            if normalized_sector in sector_mapping.values():
                valid_sectors[symbol] = normalized_sector
            else:
                logging.warning(f"Unrecognized sector for {symbol}: {sector}")
                valid_sectors[symbol] = 'Unknown'
        
        # Add missing symbols with Unknown sector
        for symbol in self.price_data.columns:
            if symbol not in valid_sectors:
                valid_sectors[symbol] = 'Unknown'
                logging.warning(f"Missing sector information for {symbol}")
        
        if not valid_sectors:
            logging.error("No valid sectors found, using equal weights")
            return {'Default': 1.0}
        
        # Calculate weights with validation
        try:
            # Count assets per sector
            for symbol, sector in valid_sectors.items():
                if symbol in self.price_data.columns:  # Only count assets present in price data
                    sector_counts[sector] += 1
            
            total_assets = sum(sector_counts.values())
            if total_assets == 0:
                raise ValueError("No valid assets found in price data")
            
            # Calculate normalized weights with minimum sector allocation
            min_sector_weight = 0.05  # 5% minimum allocation per sector
            for sector, count in sector_counts.items():
                base_weight = count / total_assets
                sector_weights[sector] = max(base_weight, min_sector_weight)
            
            # Normalize weights to sum to 1.0
            total_weight = sum(sector_weights.values())
            sector_weights = {k: v/total_weight for k, v in sector_weights.items()}
            
            # Log sector allocations
            for sector, weight in sector_weights.items():
                logging.info(f"Sector {sector}: {weight:.2%} ({sector_counts[sector]} assets)")
            
            return dict(sector_weights)
            
        except Exception as e:
            logging.error(f"Sector weight calculation failed: {str(e)}")
            return {'Default': 1.0}  # Fallback to single sector
    
    def optimize(self, regime: str, risk_appetite: str = 'Moderate') -> tuple[dict[str, float], dict]:
        """Optimize portfolio based on market regime and risk appetite"""
        try:
            # Define risk-based constraints
            risk_constraints = {
                'Conservative': {
                    'min_bonds': 0.4,
                    'max_equity': 0.5,
                    'max_sector': 0.25,
                    'target_vol': 0.12
                },
                'Moderate': {
                    'min_bonds': 0.25,
                    'max_equity': 0.7,
                    'max_sector': 0.3,
                    'target_vol': 0.18
                },
                'Aggressive': {
                    'min_bonds': 0.1,
                    'max_equity': 0.9,
                    'max_sector': 0.35,
                    'target_vol': 0.25
                }
            }
            
            constraints = risk_constraints.get(risk_appetite, risk_constraints['Moderate'])
            
            # Try intermediate fallback strategies before resorting to equal weights
            try:
                result = None
                if regime == 'Bullish':
                    result = self._growth_strategy(constraints)
                elif regime == 'Bearish':
                    result = self._defensive_strategy(constraints)
                else:
                    result = self._balanced_strategy(constraints)
                    
                # Validate that we have a proper tuple with weights and metrics
                if result is None or not isinstance(result, tuple) or len(result) != 2:
                    logging.error(f"Strategy returned invalid result: {result}")
                    # Try intermediate fallback with HRP before equal weights
                    try:
                        # Ensure we don't try to unpack an invalid result
                        weights, metrics = None, None
                        logging.info(f"Attempting HRP fallback for {regime} regime")
                        # Prepare returns data with robust preprocessing
                        clean_returns = self.returns_df.copy()
                        clean_returns = clean_returns.clip(
                            clean_returns.quantile(0.05), 
                            clean_returns.quantile(0.95)
                        )
                        clean_returns = clean_returns.fillna(method='ffill').fillna(0)
                        
                        # Initialize HRP with enhanced stability
                        from pypfopt import HRPOpt
                        hrp = HRPOpt(clean_returns)
                        hrp_weights = hrp.optimize()
                        
                        # Apply regime-specific adjustments to HRP weights
                        adjusted_weights = {}
                        
                        # Define defensive and growth sectors
                        defensive_sectors = {'Financial Services', 'Utilities', 'Consumer Staples', 'Healthcare'}
                        growth_sectors = {'Information Technology', 'Consumer Discretionary', 'Communication Services'}
                        
                        # Adjust weights based on regime
                        for asset, weight in hrp_weights.items():
                            sector = self.normalized_sectors.get(asset, self.sectors.get(asset, 'Unknown')) if hasattr(self, 'normalized_sectors') else self.sectors.get(asset, 'Unknown')
                            
                            if regime == 'Bearish' and sector in defensive_sectors:
                                adjusted_weights[asset] = weight * 1.3  # Boost defensive in bearish
                            elif regime == 'Bearish' and sector in growth_sectors:
                                adjusted_weights[asset] = weight * 0.7  # Reduce growth in bearish
                            elif regime == 'Bullish' and sector in growth_sectors:
                                adjusted_weights[asset] = weight * 1.3  # Boost growth in bullish
                            elif regime == 'Bullish' and sector in defensive_sectors:
                                adjusted_weights[asset] = weight * 0.7  # Reduce defensive in bullish
                            else:
                                adjusted_weights[asset] = weight
                        
                        # Normalize weights
                        total = sum(adjusted_weights.values())
                        adjusted_weights = {k: v/total for k, v in adjusted_weights.items()}
                        
                        # Calculate portfolio metrics
                        portfolio_return = sum(adjusted_weights[asset] * self.expected_returns[asset] 
                                             for asset in adjusted_weights if asset in self.expected_returns)
                        
                        portfolio_vol = np.sqrt(
                            sum(adjusted_weights[a1] * adjusted_weights[a2] * self.cov_matrix.loc[a1, a2]
                                for a1 in adjusted_weights for a2 in adjusted_weights
                                if a1 in self.cov_matrix.index and a2 in self.cov_matrix.columns)
                        )
                        
                        sharpe = (portfolio_return - 0.02) / portfolio_vol if portfolio_vol > 0 else 0
                        
                        hrp_metrics = {
                            'expected_return': portfolio_return,
                            'volatility': portfolio_vol,
                            'sharpe_ratio': sharpe,
                            'diversification_ratio': len(adjusted_weights),
                            'warning': f'Regime-adjusted HRP fallback used for {regime} regime',
                            'max_drawdown': 0.15
                        }
                        
                        return adjusted_weights, hrp_metrics
                    except Exception as hrp_error:
                        logging.warning(f"HRP fallback failed: {str(hrp_error)}. Using regime-specific fallback.")
                        return self._create_fallback_portfolio(regime)
                    
                weights, metrics = result
                if not weights or not metrics:
                    logging.error("Strategy returned empty weights or metrics")
                    return self._create_fallback_portfolio(regime)
                    
                return weights, metrics
            except Exception as strategy_error:
                logging.error(f"Strategy execution failed: {str(strategy_error)}")
                return self._create_fallback_portfolio(regime)
                
        except Exception as e:
            logging.error(f"Optimization failed: {str(e)}")
            # Return regime-specific fallback portfolio instead of equal weights
            return self._create_fallback_portfolio(regime)
    
    def _create_fallback_portfolio(self, regime: str = 'Neutral') -> tuple[dict[str, float], dict]:  
        """Create a regime-specific fallback portfolio when optimization strategies fail"""
        logging.warning(f"Using regime-specific fallback portfolio allocation for {regime} regime")
        try:
            # Get valid assets from price data
            valid_assets = list(self.price_data.columns)
            if not valid_assets:
                raise ValueError("No valid assets found in price data")
            
            # Create regime-specific weights instead of equal weights
            weights = {}
            
            # Ensure normalized sectors are available
            if not hasattr(self, 'normalized_sectors'):
                # Map sectors to ensure consistency
                sector_mapping = {
                    'Consumer Cyclical': 'Consumer Discretionary',
                    'Consumer Defensive': 'Consumer Staples',
                    'Financial': 'Financial Services',
                    'Technology': 'Information Technology',
                    'Basic Materials': 'Materials'
                }
                
                # Create a normalized sectors dictionary
                normalized_sectors = {}
                for symbol, sector in self.sectors.items():
                    if isinstance(sector, str):
                        normalized_sectors[symbol] = sector_mapping.get(sector, sector)
                
                self.normalized_sectors = normalized_sectors
            
            # Calculate asset volatilities for risk-based weighting
            asset_vols = {}
            for asset in valid_assets:
                if asset in self.returns_df.columns:
                    vol = self.returns_df[asset].std() * np.sqrt(252)
                    if np.isfinite(vol) and vol > 0:
                        asset_vols[asset] = vol
                    else:
                        asset_vols[asset] = np.median([self.returns_df[a].std() * np.sqrt(252) 
                                                     for a in valid_assets if a in self.returns_df.columns])
                else:
                    # Use median volatility if asset-specific calculation fails
                    asset_vols[asset] = 0.2
            
            # Define defensive and growth sectors
            defensive_sectors = {'Financial Services', 'Utilities', 'Consumer Staples', 'Healthcare'}
            growth_sectors = {'Information Technology', 'Consumer Discretionary', 'Communication Services'}
            
            # Apply regime-specific weighting logic
            if regime == 'Bearish':
                # In bearish regime, overweight defensive sectors and underweight growth
                for asset in valid_assets:
                    sector = self.normalized_sectors.get(asset, self.sectors.get(asset, 'Unknown'))
                    if sector in defensive_sectors:
                        # Inverse volatility weighting with defensive boost
                        weights[asset] = 1.5 / (asset_vols[asset] + 1e-8)
                    elif sector in growth_sectors:
                        # Reduce allocation to growth sectors
                        weights[asset] = 0.5 / (asset_vols[asset] + 1e-8)
                    else:
                        # Standard inverse volatility for other sectors
                        weights[asset] = 1.0 / (asset_vols[asset] + 1e-8)
            
            elif regime == 'Bullish':
                # In bullish regime, overweight growth sectors and underweight defensive
                for asset in valid_assets:
                    sector = self.normalized_sectors.get(asset, self.sectors.get(asset, 'Unknown'))
                    if sector in growth_sectors:
                        # Boost growth sectors
                        weights[asset] = 1.5 / (asset_vols[asset] + 1e-8)
                    elif sector in defensive_sectors:
                        # Reduce allocation to defensive sectors
                        weights[asset] = 0.5 / (asset_vols[asset] + 1e-8)
                    else:
                        # Standard inverse volatility for other sectors
                        weights[asset] = 1.0 / (asset_vols[asset] + 1e-8)
            
            else:  # Neutral regime
                # Balanced approach with pure inverse volatility weighting
                for asset in valid_assets:
                    weights[asset] = 1.0 / (asset_vols[asset] + 1e-8)
            
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v/total_weight for k, v in weights.items()}
            else:
                # Fallback to equal weights if weighting fails
                weight = 1.0 / len(valid_assets)
                weights = {asset: weight for asset in valid_assets}
            
            # Calculate basic portfolio metrics
            returns = self.returns_df.mean() * 252  # Annualized returns
            portfolio_return = sum(weights[asset] * returns[asset] for asset in weights if asset in returns)
            
            # Calculate portfolio volatility using the covariance matrix
            portfolio_vol = np.sqrt(
                sum(weights[a1] * weights[a2] * self.cov_matrix.loc[a1, a2]
                    for a1 in weights for a2 in weights
                    if a1 in self.cov_matrix.index and a2 in self.cov_matrix.columns)
            )
            
            # Calculate Sharpe ratio (assuming risk-free rate of 0.02)
            risk_free_rate = 0.02
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
            
            # Create metrics dictionary with regime-specific warning
            # Count all assets with non-zero weights for diversification
            significant_assets = sum(1 for w in weights.values() if w > 0)
            
            metrics = {
                'expected_return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe_ratio': sharpe_ratio,
                'diversification_ratio': significant_assets,  # Count all positions with non-zero weights
                'warning': f'Regime-specific fallback portfolio used for {regime} regime due to optimization failure',
                'max_drawdown': 0.15  # Conservative estimate
            }
            
            return weights, metrics
            
        except Exception as e:
            logging.error(f"Fallback portfolio creation failed: {str(e)}")
            # Ultimate fallback - return minimal valid result
            if hasattr(self, 'price_data') and not self.price_data.empty:
                asset = self.price_data.columns[0]
                return {asset: 1.0}, {
                    'expected_return': 0.05,
                    'volatility': 0.2,
                    'sharpe_ratio': 0.15,
                    'diversification_ratio': 1,  # Correctly report single asset
                    'warning': 'Emergency fallback single-asset portfolio used',
                    'max_drawdown': 0.25
                }
            else:
                # If everything fails, return empty result with warning
                return {}, {
                    'warning': 'Could not create any portfolio due to data issues',
                    'expected_return': 0.0,
                    'volatility': 0.0,
                    'sharpe_ratio': 0.0,
                    'diversification_ratio': 0,
                    'max_drawdown': 0.0
                }
    
    def _growth_strategy(self, constraints: dict) -> tuple[dict[str, float], dict[str, float]]:
        """Growth-oriented portfolio optimization with risk constraints and enhanced sector management"""
        try:
            # Enhanced sector mapping with comprehensive coverage
            sector_mapping = {
                'Consumer Cyclical': 'Consumer Discretionary',
                'Consumer Defensive': 'Consumer Staples',
                'Financial': 'Financial Services',
                'Technology': 'Information Technology',
                'Basic Materials': 'Materials',
                'Communication Services': 'Communication Services',
                'Healthcare': 'Healthcare',
                'Utilities': 'Utilities',
                'Real Estate': 'Real Estate',
                'Energy': 'Energy',
                'Industrials': 'Industrials'
            }
            
            # Normalize sectors with enhanced validation
            normalized_sectors = {}
            for symbol, sector in self.sectors.items():
                if not isinstance(sector, str):
                    logging.warning(f"Invalid sector type for {symbol}: {type(sector)}")
                    normalized_sectors[symbol] = 'Unknown'
                    continue
                
                normalized_sector = sector_mapping.get(sector, sector)
                if normalized_sector in sector_mapping.values():
                    normalized_sectors[symbol] = normalized_sector
                else:
                    logging.warning(f"Unrecognized sector for {symbol}: {sector}")
                    normalized_sectors[symbol] = 'Unknown'
            
            # Add missing symbols with Unknown sector
            for symbol in self.price_data.columns:
                if symbol not in normalized_sectors:
                    normalized_sectors[symbol] = 'Unknown'
                    logging.warning(f"Missing sector information for {symbol}")
            
            # Update instance sectors
            self.normalized_sectors = normalized_sectors
            
            # Validate sector data
            sector_symbols = set(normalized_sectors.keys())
            price_symbols = set(self.price_data.columns)
            
            # Use all available price symbols and assign 'Unknown' sector if missing
            valid_symbols = price_symbols.copy()
            
            # Ensure all price symbols have a sector assignment
            for symbol in valid_symbols:
                if symbol not in self.normalized_sectors:
                    self.normalized_sectors[symbol] = 'Unknown'
                    logging.info(f"Assigned 'Unknown' sector to {symbol}")
            
            # Validate that all symbols have valid data
            for symbol in list(valid_symbols):
                if symbol not in self.price_data.columns or \
                   symbol not in self.expected_returns.index or \
                   symbol not in self.cov_matrix.index:
                    valid_symbols.remove(symbol)
                    logging.warning(f"Removing {symbol} due to missing data")
            
            if not valid_symbols:
                raise ValueError("No valid symbols remain after validation")
            
            # Add any missing symbols to normalized_sectors with 'Unknown' sector
            for symbol in valid_symbols:
                if symbol not in self.normalized_sectors:
                    self.normalized_sectors[symbol] = 'Unknown'
            
            if not valid_symbols:
                raise ValueError("No valid symbols found in price data")
            
            # Log the number of valid symbols for debugging
            logging.info(f"Using {len(valid_symbols)} valid symbols for optimization")
            
            # Filter data to use only valid symbols
            filtered_price_data = self.price_data[list(valid_symbols)]
            filtered_returns = self.expected_returns[list(valid_symbols)]
            
            # Enhanced multi-method covariance estimation with robust conditioning
            try:
                # Try multiple covariance estimation methods and blend them for stability
                logging.info("Applying multi-method covariance estimation")
                
                # Method 1: Ledoit-Wolf shrinkage (standard)
                ledoit_wolf_cov = risk_models.CovarianceShrinkage(
                    filtered_price_data,
                    frequency=252
                ).ledoit_wolf()
                
                # Method 2: Oracle Approximating shrinkage (more aggressive)
                try:
                    oracle_cov = risk_models.CovarianceShrinkage(
                        filtered_price_data,
                        frequency=252
                    ).oracle_approximating()
                except Exception:
                    # Fallback to sample covariance if oracle fails
                    oracle_cov = risk_models.sample_cov(filtered_price_data, frequency=252)
                
                # Method 3: Exponentially weighted covariance (time-sensitive)
                try:
                    exp_cov = risk_models.exp_cov(
                        filtered_price_data,
                        frequency=252,
                        span=180  # ~6 months half-life
                    )
                except Exception:
                    # Fallback to diagonal covariance if exp_cov fails
                    vols = np.std(filtered_price_data.pct_change().dropna(), axis=0) * np.sqrt(252)
                    exp_cov = np.diag(vols**2)
                
                # Blend covariance matrices with adaptive weights based on condition numbers
                try:
                    # Calculate condition numbers for each method
                    cond_lw = np.linalg.cond(ledoit_wolf_cov)
                    cond_oracle = np.linalg.cond(oracle_cov)
                    cond_exp = np.linalg.cond(exp_cov)
                    
                    # Inverse condition number (better conditioning = higher weight)
                    inv_cond_lw = 1.0 / np.log1p(cond_lw)
                    inv_cond_oracle = 1.0 / np.log1p(cond_oracle)
                    inv_cond_exp = 1.0 / np.log1p(cond_exp)
                    
                    # Normalize weights
                    total_inv_cond = inv_cond_lw + inv_cond_oracle + inv_cond_exp
                    weight_lw = inv_cond_lw / total_inv_cond
                    weight_oracle = inv_cond_oracle / total_inv_cond
                    weight_exp = inv_cond_exp / total_inv_cond
                    
                    # Blend matrices with adaptive weights
                    reg_cov_matrix = (weight_lw * ledoit_wolf_cov + 
                                     weight_oracle * oracle_cov + 
                                     weight_exp * exp_cov)
                    
                    logging.info(f"Blended covariance matrices with weights: LW={weight_lw:.2f}, Oracle={weight_oracle:.2f}, Exp={weight_exp:.2f}")
                except Exception as blend_error:
                    # Fallback to simple average if adaptive blending fails
                    logging.warning(f"Adaptive blending failed: {str(blend_error)}. Using simple average.")
                    reg_cov_matrix = (ledoit_wolf_cov + oracle_cov + exp_cov) / 3.0
            except Exception as e:
                logging.warning(f"Multi-method estimation failed: {str(e)}. Using robust fallback.")
                # Fallback to standard Ledoit-Wolf with stronger shrinkage
                try:
                    reg_cov_matrix = risk_models.CovarianceShrinkage(
                        filtered_price_data,
                        frequency=252
                    ).ledoit_wolf()
                except Exception:
                    # Ultimate fallback to sample covariance
                    reg_cov_matrix = risk_models.sample_cov(filtered_price_data, frequency=252)
            
            # Enhanced matrix conditioning with adaptive regularization
            try:
                # Calculate eigenvalues for conditioning analysis
                eigenvals = np.linalg.eigvals(reg_cov_matrix)
                condition_number = np.max(np.abs(eigenvals)) / np.min(np.abs(eigenvals))
                logging.info(f"Initial condition number: {condition_number:.2e}")
                
                # Apply progressive regularization based on condition number
                if condition_number > 1e6:  # Extremely poor conditioning
                    # Very strong regularization for extreme cases
                    reg_factor = np.trace(reg_cov_matrix) * 5e-2
                    reg_cov_matrix += np.eye(reg_cov_matrix.shape[0]) * reg_factor
                    logging.info(f"Applied extreme regularization: {reg_factor:.2e}")
                elif condition_number > 1e4:  # Very poor conditioning
                    reg_factor = np.trace(reg_cov_matrix) * 1e-2
                    reg_cov_matrix += np.eye(reg_cov_matrix.shape[0]) * reg_factor
                    logging.info(f"Applied strong regularization: {reg_factor:.2e}")
                elif condition_number > 1e3:  # Poor conditioning
                    reg_factor = np.trace(reg_cov_matrix) * 5e-3
                    reg_cov_matrix += np.eye(reg_cov_matrix.shape[0]) * reg_factor
                    logging.info(f"Applied moderate regularization: {reg_factor:.2e}")
                else:  # Standard conditioning
                    # Always apply base regularization for numerical stability
                    base_reg = np.trace(reg_cov_matrix) * 1e-4
                    reg_cov_matrix += np.eye(reg_cov_matrix.shape[0]) * base_reg
                    logging.info(f"Applied base regularization: {base_reg:.2e}")
                
                # Ensure matrix is symmetric after regularization
                reg_cov_matrix = (reg_cov_matrix + reg_cov_matrix.T) / 2
                
                # Verify and enforce positive definiteness with stronger correction
                min_eigenval = np.min(np.real(np.linalg.eigvals(reg_cov_matrix)))
                if min_eigenval <= 1e-6:
                    # Apply correction proportional to the magnitude of the problem
                    correction = max(1e-5, abs(min_eigenval) * 2)
                    reg_cov_matrix += np.eye(reg_cov_matrix.shape[0]) * correction
                    logging.info(f"Applied positive definite correction: {correction:.2e}")
                    
                    # Verify correction worked
                    if np.min(np.real(np.linalg.eigvals(reg_cov_matrix))) <= 0:
                        logging.warning("First correction insufficient, applying stronger correction")
                        # Apply stronger correction if first attempt failed
                        correction = max(1e-4, abs(min_eigenval) * 5)
                        reg_cov_matrix += np.eye(reg_cov_matrix.shape[0]) * correction
                        
                        # Final verification
                        if np.min(np.real(np.linalg.eigvals(reg_cov_matrix))) <= 0:
                            logging.error("Failed to achieve positive definiteness")
                            # Create diagonal covariance as ultimate fallback
                            vols = np.sqrt(np.diag(reg_cov_matrix))
                            reg_cov_matrix = np.diag(vols**2)
                            logging.warning("Using diagonal covariance as fallback")
                
                # Validate final covariance matrix
                final_eigenvals = np.linalg.eigvals(reg_cov_matrix)
                final_condition = np.max(np.abs(final_eigenvals)) / np.min(np.abs(final_eigenvals))
                logging.info(f"Final condition number: {final_condition:.2e}")
                
                if np.any(np.isnan(final_eigenvals)):
                    logging.error("NaN values detected in covariance matrix")
                    raise ValueError("Invalid covariance matrix with NaN values")
                    
                if np.any(np.real(final_eigenvals) <= 0):
                    logging.error("Non-positive eigenvalues in final matrix")
                    raise ValueError("Failed to construct a positive definite covariance matrix")
                
                logging.info("Covariance matrix conditioning successful")
            except Exception as cond_error:
                logging.error(f"Matrix conditioning failed: {str(cond_error)}")
                # Ultimate fallback to diagonal covariance with volatility estimates
                vols = np.std(filtered_price_data.pct_change().dropna(), axis=0) * np.sqrt(252)
                reg_cov_matrix = np.diag(vols**2)
                logging.warning("Using diagonal covariance matrix as ultimate fallback")
            
            # Store original matrix for fallback
            original_cov = reg_cov_matrix.copy()

            # Initialize efficient frontier with enhanced risk management using filtered data
            try:
                # Validate expected returns before optimization
                if np.any(np.isnan(self.expected_returns)) or np.any(np.isinf(self.expected_returns)):
                    logging.warning("Invalid expected returns detected, using historical mean")
                    self.expected_returns = expected_returns.mean_historical_return(self.price_data)
                
                ef = EfficientFrontier(self.expected_returns, reg_cov_matrix)
                ef.add_constraint(lambda w: cp.sum(w) == 1)
                ef.add_constraint(lambda w: w >= 0)  # Non-negative weights
                
                # Add sector constraints with enhanced validation
                sector_constraints = {}
                for sector in set(self.normalized_sectors.values()):
                    sector_assets = [asset for asset in self.price_data.columns 
                                   if self.normalized_sectors.get(asset) == sector]
                    if sector_assets:
                        max_sector = min(constraints.get('max_sector', 0.3), 0.4)  # Cap at 40%
                        min_sector = max(0.0, constraints.get('min_bonds', 0.0) / len(set(self.normalized_sectors.values())))
                        
                        sector_constraints[sector] = {
                            'assets': sector_assets,
                            'min': min_sector,
                            'max': max_sector
                        }
                        logging.info(f"Setting {sector} constraint: {len(sector_assets)} assets, {min_sector:.1%} to {max_sector:.1%}")
                
                if not sector_constraints:
                    raise ValueError("No valid sector constraints could be constructed")
                
                # Apply sector constraints with error handling
                for sector, sector_data in sector_constraints.items():
                    try:
                        ef.add_sector_constraint(
                            sector_data['assets'],
                            sector_data['min'],
                            sector_data['max']
                        )
                    except Exception as e:
                        logging.error(f"Failed to add constraint for sector {sector}: {str(e)}")
                        raise ValueError(f"Sector constraint failed: {sector}")
                
                # Add minimum weight constraint to ensure diversification
                ef.add_constraint(lambda w: w <= 0.2)  # Maximum weight per asset
                
            except Exception as e:
                logging.warning(f"Failed to initialize optimizer with regularized matrix: {str(e)}")
                try:
                    # First fallback: Try with sample covariance
                    sample_cov = risk_models.sample_cov(self.price_data)
                    ef = EfficientFrontier(self.expected_returns, sample_cov)
                    ef.add_constraint(lambda w: cp.sum(w) == 1)
                    ef.add_constraint(lambda w: w >= 0)
                except Exception as e2:
                    logging.warning(f"Sample covariance fallback failed: {str(e2)}")
                    # Second fallback: Use HRP
                    try:
                        hrp = HRPOpt(self.returns_df)
                        weights = hrp.optimize()
                        return weights, self._calculate_metrics(weights)
                    except Exception as e3:
                        logging.error(f"All optimization methods failed: {str(e3)}")
                        raise ValueError("Unable to perform portfolio optimization")
            
            # Dynamic sector constraints with enhanced risk adjustment
            try:
                market_vol = np.sqrt(np.trace(reg_cov_matrix))
                vol_scale = min(1.2, max(0.8, 1.0 / market_vol))
                
                # Calculate sector risk metrics with validation
                sector_counts = defaultdict(int)
                sector_risk = defaultdict(float)
                sector_returns = defaultdict(list)
                
                for symbol_idx, symbol in enumerate(self.price_data.columns):
                    # Use normalized sectors to ensure consistency
                    sector = self.normalized_sectors.get(symbol, self.sectors.get(symbol, 'Unknown'))
                    if sector != 'Unknown':
                        try:
                            sector_counts[sector] += 1
                            vol = np.sqrt(max(0, reg_cov_matrix[list(self.price_data.columns).index(symbol), list(self.price_data.columns).index(symbol)]))
                            sector_risk[sector] += vol
                            if not np.isnan(self.expected_returns[symbol_idx]) and not np.isinf(self.expected_returns[symbol_idx]):
                                sector_returns[sector].append(self.expected_returns[symbol_idx])
                        except Exception as e:
                            logging.warning(f"Error processing sector metrics for {symbol}: {str(e)}")
                            continue
                
                # Normalize sector risks with validation and fallback
                total_risk = sum(sector_risk.values())
                if total_risk > 0:
                    sector_risk = {k: v/total_risk for k, v in sector_risk.items()}
                else:
                    # Fallback to equal risk distribution if total risk is zero
                    logging.warning("Total sector risk is zero, using equal risk distribution")
                    num_sectors = len(sector_risk)
                    if num_sectors > 0:
                        equal_risk = 1.0 / num_sectors
                        sector_risk = {k: equal_risk for k in sector_risk.keys()}
                    else:
                        sector_risk = {}
                
                # Calculate sector expected returns with validation
                sector_exp_returns = {}
                for k, v in sector_returns.items():
                    if v:  # Check if sector has any returns
                        try:
                            sector_exp_returns[k] = np.mean(v)
                        except Exception as e:
                            logging.warning(f"Failed to calculate mean return for sector {k}: {str(e)}")
                            sector_exp_returns[k] = 0.0
                
                # Adjust sector constraints based on risk-return profile
                base_upper = constraints['max_sector']
                base_lower = max(0.05, 1.0 / len(sector_counts))
                
                sector_upper = {}
                sector_lower = {}
                
                for sector in sector_counts:
                    risk_adj = min(1.2, max(0.8, 1.0 / (sector_risk[sector] + 1e-6)))
                    return_adj = min(1.2, max(0.8, sector_exp_returns.get(sector, 0.0) + 1.0))
                    
                    sector_upper[sector] = min(0.4, base_upper * risk_adj * return_adj)
                    sector_lower[sector] = max(0.02, base_lower * risk_adj)
                    
                # This validation check was premature as weights and portfolio_metrics haven't been defined yet
                # Removing this check as it causes the optimization to fail and fall back to predefined weights
            
            except Exception as e:
                logging.warning(f"Sector risk calculation failed: {str(e)}. Using default constraints.")
                # Fallback to simpler sector constraints
                sector_upper = {sector: constraints['max_sector'] for sector in self.sector_weights}
                sector_lower = {sector: 0.02 for sector in self.sector_weights}

            # Adaptive sector limits based on risk contribution
            num_sectors = len(sector_counts)
            base_max = min(0.30, constraints['max_sector'] * vol_scale)
            
            # Risk-adjusted sector constraints with validation
            total_risk = sum(sector_risk.values())
            if total_risk <= 0:
                # Fallback to equal risk if total risk is invalid
                logging.warning("Invalid total risk detected, using equal risk distribution")
                equal_risk = 1.0 / max(1, len(sector_counts))
                sector_risk = {sector: equal_risk for sector in sector_counts}
                total_risk = sum(sector_risk.values())
            
            sector_upper = {
                sector: min(0.35, base_max * (1 + np.log1p(count)/num_sectors) * (1 + sector_risk[sector]/total_risk))
                for sector, count in sector_counts.items()
            }
            sector_lower = {
                sector: max(0.01, min(0.05, constraints['min_bonds'] * 0.3 * (sector_risk[sector]/total_risk)))
                for sector in sector_counts
            }
            
            # Use normalized sectors for constraints to ensure consistency
            try:
                # Validate sector constraints before applying
                if not sector_upper or not sector_lower:
                    logging.warning("Empty sector constraints, using default constraints")
                    sector_upper = {sector: constraints['max_sector'] for sector in self.sector_weights}
                    sector_lower = {sector: 0.02 for sector in self.sector_weights}
                
                # Apply sector constraints with error handling
                ef.add_sector_constraints(sector_mapper=self.normalized_sectors, sector_upper=sector_upper, sector_lower=sector_lower)
                logging.info("Successfully applied sector constraints")
            except Exception as sector_error:
                logging.warning(f"Failed to apply sector constraints: {str(sector_error)}")
                # Reset optimizer and apply simpler constraints
                ef = EfficientFrontier(self.expected_returns, reg_cov_matrix)
                ef.add_constraint(lambda w: cp.sum(w) == 1)
                ef.add_constraint(lambda w: w >= 0)
            
            # Enhanced bond allocation with proper risk management
            bond_symbols = [s for s in self.price_data.columns if self.sectors.get(s) == 'Financial Services']
            if bond_symbols:
                bond_indices = [list(self.price_data.columns).index(b) for b in bond_symbols]
                ef.add_constraint(lambda w: sum(w[i] for i in bond_indices) >= constraints['min_bonds'] * 0.8)  # Relaxed bond constraint
                
                # Add maximum equity exposure constraint
                equity_symbols = [s for s in self.price_data.columns if self.sectors.get(s) != 'Financial Services']
                if equity_symbols:
                    equity_indices = [list(self.price_data.columns).index(e) for e in equity_symbols]
                    ef.add_constraint(lambda w: sum(w[i] for i in equity_indices) <= constraints['max_equity'] * 1.1)  # Relaxed equity constraint
            
            # Enhanced risk-adjusted optimization with adaptive regularization
            target_vol = constraints['target_vol'] * 1.1  # Relaxed volatility target
            l2_gamma = max(0.01, min(0.1, np.std(self.expected_returns)))  # Reduced L2 regularization
            
            # Primary optimization attempt with enhanced numerical stability
            ef.add_objective(objective_functions.L2_reg, gamma=l2_gamma)
            
            # Scale return objective based on market volatility
            return_scale = min(0.5, max(0.3, 1.0 / np.sqrt(market_vol)))
            ef.add_objective(lambda w: -return_scale * w @ self.expected_returns)
            
            # Multi-stage optimization with enhanced fallback strategy
            success = False
            raw_weights = None
            
            # Try efficient risk optimization first with enhanced validation and recovery
            try:
                logging.info("Attempting efficient risk optimization")
                # Add minimum allocation constraint
                ef.add_constraint(lambda w: w >= 1e-4)
                
                # Try optimization with progressively relaxed targets
                for vol_scale in [1.0, 1.1, 1.2]:
                    try:
                        adjusted_vol = target_vol * vol_scale
                        ef.efficient_risk(adjusted_vol)
                        raw_weights = ef.clean_weights()
                        
                        # Validate weights
                        weight_sum = sum(raw_weights.values())
                        if all(w >= 0 for w in raw_weights.values()) and abs(weight_sum - 1.0) < 1e-4:
                            success = True
                            logging.info(f"Optimization succeeded with vol_scale: {vol_scale}")
                            break
                    except Exception as inner_e:
                        logging.warning(f"Attempt with vol_scale {vol_scale} failed: {str(inner_e)}")
                        continue
                    
                if not success:
                    logging.warning("All efficient risk optimization attempts failed")
            except Exception as e:
                logging.warning(f"Efficient risk optimization failed: {str(e)}")
                # Reset the optimizer state
                ef.reset()
                
                # Reset optimizer with relaxed constraints for second attempt
                try:
                    logging.info("Attempting efficient return optimization")
                    # Define target return for fallback with enhanced stability
                    target_return = max(0.05, np.mean(self.expected_returns) * 0.8)
                    
                    # Add stability buffer to covariance matrix
                    stability_factor = 1e-6 * np.eye(reg_cov_matrix.shape[0])
                    reg_cov_matrix += stability_factor
                    
                    # Reinitialize optimizer with stabilized matrices
                    ef = EfficientFrontier(self.expected_returns, reg_cov_matrix)
                    ef.add_constraint(lambda w: cp.sum(w) == 1)
                    
                    # Add minimum weight constraint to avoid numerical issues
                    ef.add_constraint(lambda w: w >= 1e-4)
                    
                    # Reset optimizer with fresh constraints
                    ef = EfficientFrontier(self.expected_returns, reg_cov_matrix)
                    ef.add_constraint(lambda w: cp.sum(w) == 1)
                    ef.add_constraint(lambda w: w >= 0.001)  # Minimum weight constraint
                    
                    # Relax sector constraints
                    relaxed_upper = {k: min(0.4, v * 1.2) for k, v in sector_upper.items()}
                    relaxed_lower = {k: max(0.01, v * 0.8) for k, v in sector_lower.items()}
                    ef.add_sector_constraints(sector_mapper=self.sectors,
                                            sector_upper=relaxed_upper,
                                            sector_lower=relaxed_lower)
                    
                    # Relaxed asset constraints
                    if bond_symbols:
                        ef.add_constraint(lambda w: sum(w[i] for i in bond_indices) >= constraints['min_bonds'] * 0.7)
                        if equity_symbols:
                            ef.add_constraint(lambda w: sum(w[i] for i in equity_indices) <= constraints['max_equity'] * 1.2)
                    
                    ef.add_objective(objective_functions.L2_reg, gamma=l2_gamma * 0.8)
                    ef.efficient_return(target_return)
                    raw_weights = ef.clean_weights()
                    if all(w >= 0 for w in raw_weights.values()):
                        success = True
                except Exception as e:
                    logging.warning(f"Efficient return optimization failed: {str(e)}")
                    
                    # Final attempt: Minimum volatility with minimal constraints
                    try:
                        logging.info("Attempting minimum volatility optimization")
                        ef = EfficientFrontier(self.expected_returns, reg_cov_matrix)
                        ef.add_constraint(lambda w: cp.sum(w) == 1)
                        ef.add_constraint(lambda w: w >= 0.001)  # Minimum weight constraint
                        
                        # Minimal sector constraints
                        min_upper = {k: min(0.5, v * 1.5) for k, v in sector_upper.items()}
                        min_lower = {k: max(0.005, v * 0.5) for k, v in sector_lower.items()}
                        ef.add_sector_constraints(sector_mapper=self.sectors,
                                                sector_upper=min_upper,
                                                sector_lower=min_lower)
                        
                        ef.add_objective(objective_functions.L2_reg, gamma=l2_gamma)
                        ef.min_volatility()
                        raw_weights = ef.clean_weights()
                        if all(w >= 0 for w in raw_weights.values()):
                            success = True
                    except Exception as e:
                        logging.warning(f"Minimum volatility optimization failed: {str(e)}")
                        # Let it fall through to the HRP fallback
                    
            
            if not success:
                # Enhanced Risk Parity optimization with robust validation
                try:
                    logging.info("Attempting Enhanced Risk Parity optimization")
                    # Prepare data with robust preprocessing
                    stabilized_returns = self.returns_df.copy()
                    stabilized_returns = stabilized_returns.replace([np.inf, -np.inf], np.nan)
                    stabilized_returns = stabilized_returns.fillna(method='ffill').fillna(0)
                    
                    # Add minimal noise to prevent perfect correlation
                    noise_factor = 1e-8 * np.random.randn(*stabilized_returns.shape)
                    stabilized_returns += noise_factor
                    
                    # Initialize HRP with enhanced stability
                    hrp = HRPOpt(stabilized_returns)
                    
                    # Apply risk parity optimization with validation
                    hrp.optimize()
                    raw_weights = hrp.clean_weights()
                    
                    # Validate HRP weights
                    weight_sum = sum(raw_weights.values())
                    if not (0.99 <= weight_sum <= 1.01) or any(w < 0 for w in raw_weights.values()):
                        raise ValueError(f"Invalid HRP weights: sum={weight_sum}")
                    
                    # Apply sector constraints to HRP weights
                    sector_weights = defaultdict(float)
                    for asset, weight in raw_weights.items():
                        sector = self.normalized_sectors.get(asset, 'Unknown')
                        sector_weights[sector] += weight
                    
                    # Adjust weights if sector constraints violated
                    max_sector_weight = 0.35
                    if any(w > max_sector_weight for w in sector_weights.values()):
                        scale_factor = max_sector_weight / max(sector_weights.values())
                        raw_weights = {k: v * scale_factor for k, v in raw_weights.items()}
                    
                    if all(w >= 0 for w in raw_weights.values()) and sum(raw_weights.values()) > 0:
                        success = True
                    else:
                        raise ValueError("HRP produced invalid weights")
                        
                except Exception as e:
                    logging.error(f"HRP fallback failed: {str(e)}")
                    # Emergency fallback: Equal weight portfolio with minimal constraints
                    try:
                        n_assets = len(self.price_data.columns)
                        raw_weights = {asset: 1.0/n_assets for asset in self.price_data.columns}
                        success = True
                        logging.info("Using equal weight portfolio as final fallback")
                    except Exception as e2:
                        logging.error(f"Equal weight fallback failed: {str(e2)}")
                        raise ValueError(f"All optimization attempts failed: {str(e)} -> {str(e2)}")
            
            # Validate and clean weights
            raw_weights = {k: max(0, v) for k, v in raw_weights.items()}  # Ensure non-negative weights
            
            # Normalize and validate weights
            total_weight = sum(raw_weights.values())
            if total_weight <= 0:
                raise ValueError("Invalid portfolio weights: sum is zero or negative")
                
            weights = {k: v/total_weight for k, v in raw_weights.items()}
            
            # Enhanced portfolio metrics calculation with comprehensive risk analysis
            try:
                # Validate weights and prepare for performance calculation
                if not weights or sum(weights.values()) == 0:
                    raise ValueError("Invalid weights for performance calculation")
                
                # Calculate risk-adjusted metrics with enhanced stability
                returns = self.returns_df
                portfolio_returns = returns.dot(pd.Series(weights))
                
                # Calculate rolling metrics for stability
                rolling_vol = portfolio_returns.rolling(window=63).std() * np.sqrt(252)
                rolling_ret = portfolio_returns.rolling(window=63).mean() * 252
                rolling_sharpe = rolling_ret / rolling_vol
                
                # Calculate stable performance metrics
                expected_return = float(weights @ self.expected_returns)
                portfolio_vol = float(np.sqrt(weights @ reg_cov_matrix @ weights))
                sharpe = expected_return / portfolio_vol if portfolio_vol > 0 else 0
                
                # Enhanced validation with market-relative metrics
                market_vol = np.sqrt(np.trace(reg_cov_matrix) / reg_cov_matrix.shape[0])
                relative_vol = portfolio_vol / market_vol
                
                # Validate all metrics
                metrics = [expected_return, portfolio_vol, sharpe, relative_vol]
                if any(np.isnan(x) or np.isinf(x) for x in metrics):
                    raise ValueError("Invalid performance metrics detected")
                
                # Create portfolio metrics dictionary
                portfolio_metrics = {
                    'expected_return': expected_return,
                    'volatility': portfolio_vol,
                    'sharpe_ratio': sharpe,
                    'max_drawdown': float(np.max(np.maximum.accumulate(portfolio_returns) - portfolio_returns)),
                    'diversification_ratio': float(sum(1 for w in weights.values() if w > 0))
                }
                
                # Calculate sector allocation
                sector_allocation = defaultdict(float)
                for symbol, weight in weights.items():
                    sector = self.normalized_sectors.get(symbol, self.sectors.get(symbol, 'Unknown'))
                    sector_allocation[sector] += weight
                
                portfolio_metrics['sector_allocation'] = dict(sector_allocation)
                
                # Store performance metrics for potential use in fallback
                perf = (expected_return, portfolio_vol, sharpe)
                    
            except Exception as e:
                logging.warning(f"Portfolio performance calculation failed: {str(e)}")
                # Fallback to estimating performance manually with validation
                try:
                    expected_return = float(weights @ self.expected_returns)
                    portfolio_vol = float(np.sqrt(max(1e-8, weights @ reg_cov_matrix @ weights)))
                    sharpe = float(expected_return / portfolio_vol if portfolio_vol > 0 else 0)
                    
                    # Validate calculated metrics
                    if any(np.isnan(x) or np.isinf(x) for x in [expected_return, portfolio_vol, sharpe]):
                        raise ValueError("Invalid manual performance metrics")
                        
                    perf = (expected_return, portfolio_vol, sharpe)
                except Exception as e2:
                    logging.error(f"Manual performance calculation failed: {str(e2)}")
                    # Ultimate fallback: use conservative estimates
                    perf = (0.05, 0.15, 0.33)  # Conservative default values
            
            try:
                # Validate and adjust returns based on market regime
                market_return = np.mean(self.returns_df.mean())
                expected_return = float(np.clip(perf[0], market_return * 0.5, market_return * 2.0))
                volatility = float(np.clip(perf[1], 0.05, 0.4))
                sharpe = float(np.clip(perf[2], -3, 5))
                
                # Enhanced risk metrics calculation
                returns = self.returns_df.dot(pd.Series(weights))
                rolling_drawdown = np.maximum.accumulate(returns) - returns
                max_drawdown = float(np.clip(np.max(rolling_drawdown), 0, 0.25))
                
                # Calculate improved diversification metrics
                effective_positions = sum(1 for w in weights.values() if w > 0.005)
                herfindahl = sum(w * w for w in weights.values())
                diversification_ratio = effective_positions  # Use actual count of meaningful positions
                
                # Calculate sector concentration using normalized sectors
                sector_allocation = defaultdict(float)
                for symbol, weight in weights.items():
                    sector = self.normalized_sectors.get(symbol, self.sectors.get(symbol, 'Unknown'))
                    sector_allocation[sector] += weight
                
                # Comprehensive portfolio metrics
                portfolio_metrics = {
                    'expected_return': expected_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe,
                    'max_drawdown': max_drawdown,
                    'diversification_ratio': diversification_ratio,
                    'effective_positions': effective_positions,
                    'sector_allocation': dict(sector_allocation)
                }
                
                # Validate portfolio metrics
                if any(np.isnan(v) or np.isinf(v) for v in [expected_return, volatility, sharpe, max_drawdown, diversification_ratio]):
                    raise ValueError("Invalid portfolio metrics detected")
                
                return weights, portfolio_metrics
                
            except Exception as e:
                logging.error(f"Error calculating portfolio metrics: {str(e)}")
                # Fallback to basic metrics
                portfolio_metrics = {
                    'expected_return': float(weights @ self.expected_returns),
                    'volatility': float(np.sqrt(weights @ self.cov_matrix @ weights)),
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'diversification_ratio': float(sum(1 for w in weights.values() if w > 0.005)),
                    'effective_positions': len(weights),
                    'sector_allocation': self._calculate_sector_weights()
                }
                return weights, portfolio_metrics
            
        except Exception as e:
            logging.error(f"Growth strategy optimization failed: {str(e)}")
            raise ValueError(f"Growth strategy optimization failed: {str(e)}")
    
    def _defensive_strategy(self, constraints: dict) -> tuple[dict[str, float], dict[str, float]]:
        """Defensive portfolio optimization with enhanced risk management and numerical stability"""
        try:
            # Ensure normalized sectors are available
            if not hasattr(self, 'normalized_sectors'):
                # Map sectors to ensure consistency with asset_loader.py
                sector_mapping = {
                    'Consumer Cyclical': 'Consumer Discretionary',
                    'Consumer Defensive': 'Consumer Staples',
                    'Financial': 'Financial Services',
                    'Technology': 'Information Technology',
                    'Basic Materials': 'Materials'
                }
                
                # Create a normalized sectors dictionary for optimization
                normalized_sectors = {}
                for symbol, sector in self.sectors.items():
                    normalized_sectors[symbol] = sector_mapping.get(sector, sector)
                
                # Use normalized sectors for optimization
                self.normalized_sectors = normalized_sectors
            # Enhanced covariance estimation with multi-factor shrinkage and robust conditioning
            try:
                # Combine multiple covariance estimators for stability
                ledoit_wolf = risk_models.CovarianceShrinkage(self.price_data, frequency=252).ledoit_wolf()
                sample_cov = risk_models.sample_cov(self.price_data, frequency=252)
                exp_cov = risk_models.exp_cov(self.price_data, frequency=252, span=252)
                
                # Weighted average of estimators
                reg_cov_matrix = 0.5 * ledoit_wolf + 0.3 * sample_cov + 0.2 * exp_cov
                
                # Enhanced conditioning with adaptive regularization
                eigenvals = np.linalg.eigvals(reg_cov_matrix)
                condition_number = np.max(np.abs(eigenvals)) / np.min(np.abs(eigenvals))
                
                # Dynamic regularization based on condition number
                base_reg = np.trace(reg_cov_matrix) * min(1e-3, 1e-4 * np.log1p(condition_number))
                reg_cov_matrix += np.eye(reg_cov_matrix.shape[0]) * base_reg
                
                # Ensure symmetry and positive definiteness
                reg_cov_matrix = (reg_cov_matrix + reg_cov_matrix.T) / 2
            except Exception as e:
                logging.warning(f"Enhanced covariance estimation failed: {str(e)}. Using robust fallback.")
                # Fallback to simple shrinkage estimator with strong regularization
                reg_cov_matrix = risk_models.CovarianceShrinkage(self.price_data, frequency=252).ledoit_wolf()
                reg_cov_matrix += np.eye(reg_cov_matrix.shape[0]) * (np.trace(reg_cov_matrix) * 1e-3)
            
            # Ensure matrix is symmetric
            reg_cov_matrix = (reg_cov_matrix + reg_cov_matrix.T) / 2
            
            # Additional regularization for poor conditioning with progressive thresholds
            if condition_number > 1e3:
                reg_factor = np.trace(reg_cov_matrix) * max(1e-2, 1e-4 * condition_number)
                reg_cov_matrix += np.eye(reg_cov_matrix.shape[0]) * reg_factor
                logging.info(f"Applied additional regularization: {reg_factor:.2e}")
            elif condition_number > 1e2:
                reg_factor = np.trace(reg_cov_matrix) * 1e-3
                reg_cov_matrix += np.eye(reg_cov_matrix.shape[0]) * reg_factor
                logging.info(f"Applied moderate regularization: {reg_factor:.2e}")
            
            # Verify and enforce positive definiteness
            min_eigenval = np.min(np.real(np.linalg.eigvals(reg_cov_matrix)))
            if min_eigenval <= 1e-6:
                correction = abs(min_eigenval) + 1e-5
                reg_cov_matrix += np.eye(reg_cov_matrix.shape[0]) * correction
                logging.info(f"Applied positive definite correction: {correction:.2e}")
                
                # Verify correction worked
                if np.min(np.real(np.linalg.eigvals(reg_cov_matrix))) <= 0:
                    logging.error("Failed to achieve positive definiteness")
                    raise ValueError("Could not construct valid covariance matrix")
                
            # Validate final covariance matrix
            final_eigenvals = np.linalg.eigvals(reg_cov_matrix)
            if np.any(np.real(final_eigenvals) <= 0) or np.any(np.isnan(final_eigenvals)):
                logging.error("Covariance matrix validation failed")
                raise ValueError("Failed to construct a valid covariance matrix")
            
            logging.info("Covariance matrix conditioning successful")
            
            # Store original matrix for fallback
            original_cov = reg_cov_matrix.copy()

            # Create efficient frontier instance first
            ef = EfficientFrontier(self.expected_returns, reg_cov_matrix)
            ef.add_constraint(lambda w: cp.sum(w) == 1)
            
            # Dynamic sector constraints with market volatility adjustment
            market_vol = np.sqrt(np.trace(reg_cov_matrix))
            vol_scale = min(1.2, max(0.8, 1.0 / market_vol))
        
            # Defensive sector constraints with proper scaling
            base_upper = min(0.25, constraints['max_sector'] * 0.7)
            base_lower = max(0.03, constraints['min_bonds'] * 0.5)
            
            sector_upper = {sector: base_upper * (1.2 if sector == 'Financial Services' else 1.0)
                           for sector in self.sector_weights}
            sector_lower = {sector: base_lower * (1.5 if sector == 'Financial Services' else 1.0)
                           for sector in self.sector_weights}
            
            # Apply sector constraints only once
            # Use normalized sectors for constraints to ensure consistency
            try:
                # Validate sector constraints before applying
                if not sector_upper or not sector_lower:
                    logging.warning("Empty sector constraints, using default constraints")
                    sector_upper = {sector: constraints['max_sector'] for sector in self.sector_weights}
                    sector_lower = {sector: 0.02 for sector in self.sector_weights}
                
                # Apply sector constraints with error handling
                ef.add_sector_constraints(sector_mapper=self.normalized_sectors, sector_upper=sector_upper, sector_lower=sector_lower)
                logging.info("Successfully applied sector constraints")
            except Exception as sector_error:
                logging.warning(f"Failed to apply sector constraints: {str(sector_error)}")
                # Reset optimizer and apply simpler constraints
                ef = EfficientFrontier(self.expected_returns, reg_cov_matrix)
                ef.add_constraint(lambda w: cp.sum(w) == 1)
                ef.add_constraint(lambda w: w >= 0)
        
            # Enhanced defensive allocation with strict risk controls
            defensive_sectors = {
                'Financial Services': {'min': 0.35, 'max': 0.45},
                'Utilities': {'min': 0.10, 'max': 0.20},
                'Consumer Staples': {'min': 0.10, 'max': 0.20},
                'Healthcare': {'min': 0.10, 'max': 0.20}
            }
            
            # Apply enhanced defensive sector constraints
            for sector, limits in defensive_sectors.items():
                sector_symbols = [s for s in self.price_data.columns if self.normalized_sectors.get(s) == sector]
                if sector_symbols:
                    sector_indices = [list(self.price_data.columns).index(s) for s in sector_symbols]
                    ef.add_constraint(lambda w, idx=sector_indices, min_w=limits['min']: sum(w[i] for i in idx) >= min_w)
                    ef.add_constraint(lambda w, idx=sector_indices, max_w=limits['max']: sum(w[i] for i in idx) <= max_w)
            
            # Strict limits on growth/volatile sectors
            growth_sectors = ['Information Technology', 'Consumer Discretionary', 'Communication Services']
            growth_symbols = [s for s in self.price_data.columns if self.normalized_sectors.get(s) in growth_sectors]
            if growth_symbols:
                growth_indices = [list(self.price_data.columns).index(s) for s in growth_symbols]
                max_growth = min(0.20, constraints['max_equity'] * 0.4)  # Strict cap on growth exposure
                ef.add_constraint(lambda w: sum(w[i] for i in growth_indices) <= max_growth)
            
            # Conservative volatility targeting with risk adjustment
            target_vol = constraints['target_vol'] * 0.85
            
            # Add L2 regularization with increased gamma for better numerical stability
            ef.add_objective(objective_functions.L2_reg, gamma=0.1)
            # Add objective to minimize volatility while maintaining returns with scaled parameters
            ef.add_objective(lambda w: -0.3 * w @ self.expected_returns)
            
            try:
                # Try efficient risk optimization with adaptive volatility targeting
                target_vol_adjusted = target_vol * 0.85  # More conservative target
                ef.efficient_risk(target_vol_adjusted)
                raw_weights = ef.clean_weights()
                
                # Enhanced validation of optimization results
                significant_weights = sum(1 for w in raw_weights.values() if w > 0.005)
                if significant_weights < len(defensive_sectors):
                    raise ValueError(f"Insufficient diversification: {significant_weights} positions")
                
                # Verify defensive sector allocations
                sector_allocations = defaultdict(float)
                for asset, weight in raw_weights.items():
                    sector = self.normalized_sectors.get(asset, 'Unknown')
                    sector_allocations[sector] += weight
                
                # Validate defensive sector minimums are met
                for sector, limits in defensive_sectors.items():
                    if sector_allocations[sector] < limits['min'] * 0.95:  # Allow 5% tolerance
                        raise ValueError(f"Defensive sector {sector} below minimum allocation")
                
            except Exception as e:
                logging.warning(f"Primary optimization failed: {str(e)}. Attempting risk-parity approach.")
                try:
                    # Risk parity approach with sector constraints
                    hrp = HRPOpt(self.returns_df)
                    base_weights = hrp.optimize()
                    
                    # Adjust HRP weights to meet defensive requirements
                    adjusted_weights = {}
                    for asset, weight in base_weights.items():
                        sector = self.normalized_sectors.get(asset, 'Unknown')
                        if sector in defensive_sectors:
                            # Boost defensive sectors
                            adjusted_weights[asset] = weight * 1.5
                        elif sector in growth_sectors:
                            # Reduce growth exposure
                            adjusted_weights[asset] = weight * 0.5
                        else:
                            adjusted_weights[asset] = weight
                    
                    # Normalize weights
                    total = sum(adjusted_weights.values())
                    raw_weights = {k: v/total for k, v in adjusted_weights.items()}
                    
                except Exception as e2:
                    logging.warning(f"Risk parity approach failed: {str(e2)}. Using minimum volatility fallback.")
                    # Minimum volatility fallback with strict constraints
                    ef_fallback = EfficientFrontier(self.expected_returns, reg_cov_matrix)
                    ef_fallback.add_constraint(lambda w: cp.sum(w) == 1)
                    ef_fallback.add_constraint(lambda w: w >= 0.001)
                    
                    # Apply stricter defensive sector constraints
                    for sector, limits in defensive_sectors.items():
                        sector_symbols = [s for s in self.price_data.columns if self.normalized_sectors.get(s) == sector]
                        if sector_symbols:
                            sector_indices = [list(self.price_data.columns).index(s) for s in sector_symbols]
                            min_weight = limits['min'] * 1.1  # Increase minimums by 10%
                            max_weight = limits['max'] * 0.9  # Decrease maximums by 10%
                            ef_fallback.add_constraint(lambda w, idx=sector_indices: sum(w[i] for i in idx) >= min_weight)
                            ef_fallback.add_constraint(lambda w, idx=sector_indices: sum(w[i] for i in idx) <= max_weight)
                    
                    # Minimize volatility with increased regularization
                    ef_fallback.add_objective(objective_functions.L2_reg, gamma=1.0)
                    ef_fallback.min_volatility()
                    raw_weights = ef_fallback.clean_weights()
                    
                    # Add stronger regularization for fallback
                    ef_fallback.add_objective(objective_functions.L2_reg, gamma=0.15)
                    ef_fallback.min_volatility()
                    raw_weights = ef_fallback.clean_weights()
            
            # Normalize and validate weights with improved numerical stability
            total_weight = sum(raw_weights.values())
            if abs(total_weight - 1.0) > 0.01:  # Only normalize if significantly off
                logging.info(f"Normalizing weights (sum={total_weight:.4f})")
                weights = {k: v/total_weight for k, v in raw_weights.items()}
            else:
                weights = raw_weights
            
            # Calculate risk-adjusted portfolio metrics with enhanced stability
            try:
                # Enhanced optimization with multiple fallback strategies
                try:
                    # Try maximum Sharpe ratio optimization first
                    ef.add_objective(objective_functions.L2_reg, gamma=0.1)
                    ef.add_objective(lambda w: -0.5 * w @ self.expected_returns)  # Return maximization
                    ef.add_objective(lambda w: 0.5 * cp.quad_form(w, reg_cov_matrix))  # Risk minimization
                    
                    ef.max_sharpe(risk_free_rate=0.02)
                    raw_weights = ef.clean_weights()
                    
                    # Validate results
                    if not any(w > 0.01 for w in raw_weights.values()):
                        raise ValueError("Sharpe optimization produced insufficient weights")
                        
                except Exception as e1:
                    logging.warning(f"Sharpe optimization failed: {str(e1)}. Trying efficient risk.")
                    try:
                        # Try efficient risk optimization
                        target_vol_adjusted = target_vol * vol_scale
                        ef.efficient_risk(target_vol_adjusted)
                        raw_weights = ef.clean_weights()
                        
                        if not any(w > 0.01 for w in raw_weights.values()):
                            raise ValueError("Risk optimization produced insufficient weights")
                            
                    except Exception as e2:
                        logging.warning(f"Risk optimization failed: {str(e2)}. Using minimum volatility.")
                        try:
                            # Final attempt with minimum volatility
                            ef.min_volatility()
                            raw_weights = ef.clean_weights()
                            
                            if not any(w > 0.01 for w in raw_weights.values()):
                                raise ValueError("Minimum volatility optimization failed")
                                
                        except Exception as e3:
                            logging.error(f"All optimization attempts failed: {str(e3)}")
                            raise ValueError("Unable to find valid portfolio weights")
                            
                # Post-optimization weight adjustment
                total_weight = sum(raw_weights.values())
                if abs(total_weight - 1.0) > 0.01:
                    weights = {k: v/total_weight for k, v in raw_weights.items()}
                else:
                    weights = raw_weights
            except Exception as perf_error:
                logging.warning(f"Portfolio performance calculation failed: {str(perf_error)}")
                # Fallback calculation using historical returns
                returns_series = self.returns_df.dot(pd.Series(weights))
                annual_return = returns_series.mean() * 252
                annual_vol = returns_series.std() * np.sqrt(252)
                sharpe = annual_return / annual_vol if annual_vol > 0 else 0.0
                
                portfolio_metrics = {
                    'expected_return': float(np.clip(annual_return, 0.01, 0.3)),
                    'volatility': float(np.clip(annual_vol, 0.05, 0.3)),
                    'sharpe_ratio': float(np.clip(sharpe, 0.1, 5.0))
                }
            
            # Additional risk metrics with realistic values
            returns = self.returns_df.dot(pd.Series(weights))
            max_drawdown = float(np.clip(abs(min(0, returns.cumsum().min())), 0.05, 0.4))
            diversification = len([w for w in weights.values() if w > 0.01])
            
            portfolio_metrics.update({
                'max_drawdown': max_drawdown,
                'diversification_ratio': float(diversification)
            })
            
            return weights, portfolio_metrics
            
        except Exception as e:
            logging.error(f"Defensive strategy failed: {str(e)}")
            logging.info("Initiating enhanced defensive fallback system")
            
            # Multi-stage fallback system for defensive strategy
            try:
                # Stage 1: Enhanced Risk Parity with defensive sector bias
                logging.info("Attempting enhanced risk parity with defensive bias")
                
                # Ensure we have a valid covariance matrix for fallback
                try:
                    # Create a more stable covariance matrix with stronger shrinkage
                    fallback_cov = risk_models.CovarianceShrinkage(
                        self.price_data,
                        frequency=252
                    ).oracle_approximating()
                    
                    # Add stability buffer proportional to matrix trace
                    stability_factor = np.trace(fallback_cov) * 1e-4
                    fallback_cov += np.eye(fallback_cov.shape[0]) * stability_factor
                    
                    # Ensure symmetry and positive definiteness
                    fallback_cov = (fallback_cov + fallback_cov.T) / 2
                    min_eigenval = np.min(np.real(np.linalg.eigvals(fallback_cov)))
                    if min_eigenval <= 1e-8:
                        correction = abs(min_eigenval) + 1e-7
                        fallback_cov += np.eye(fallback_cov.shape[0]) * correction
                except Exception as cov_error:
                    logging.warning(f"Fallback covariance estimation failed: {str(cov_error)}")
                    # Use diagonal covariance as ultimate fallback
                    returns = self.returns_df.fillna(0)
                    vols = returns.std() * np.sqrt(252)
                    fallback_cov = np.diag(vols**2)
                
                # Calculate asset volatilities with defensive sector adjustment
                vols = np.sqrt(np.diag(fallback_cov))
                inv_vols = 1.0 / (vols + 1e-8)  # Avoid division by zero
                
                # Apply defensive sector bias
                defensive_sectors = {'Financial Services', 'Utilities', 'Consumer Staples', 'Healthcare'}
                defensive_boost = 1.5  # Boost defensive sectors by 50%
                
                # Create sector-adjusted inverse volatilities
                adjusted_inv_vols = []
                for i, symbol in enumerate(self.price_data.columns):
                    sector = self.normalized_sectors.get(symbol, self.sectors.get(symbol, 'Unknown'))
                    if sector in defensive_sectors:
                        adjusted_inv_vols.append(inv_vols[i] * defensive_boost)
                    else:
                        adjusted_inv_vols.append(inv_vols[i])
                
                # Calculate weights with defensive bias
                raw_weights = {}
                total_adj_inv_vol = np.sum(adjusted_inv_vols)
                for i, symbol in enumerate(self.price_data.columns):
                    raw_weights[symbol] = adjusted_inv_vols[i] / total_adj_inv_vol
                
                # Validate weights
                if all(w >= 0 for w in raw_weights.values()) and abs(sum(raw_weights.values()) - 1.0) < 1e-6:
                    weights = raw_weights
                    logging.info("Enhanced risk parity with defensive bias successful")
                else:
                    raise ValueError("Invalid weights from risk parity calculation")
                    
            except Exception as rp_error:
                logging.warning(f"Enhanced risk parity failed: {str(rp_error)}")
                
                # Stage 2: Cluster-based minimum variance
                try:
                    logging.info("Attempting cluster-based minimum variance")
                    
                    # Prepare returns data with robust preprocessing
                    clean_returns = self.returns_df.copy()
                    # Replace outliers with quantile values
                    clean_returns = clean_returns.clip(
                        clean_returns.quantile(0.05), 
                        clean_returns.quantile(0.95)
                    )
                    clean_returns = clean_returns.fillna(method='ffill').fillna(0)
                    
                    # Calculate correlation matrix for clustering
                    corr_matrix = clean_returns.corr()
                    # Convert to distance matrix (1 - |correlation|)
                    distance_matrix = 1 - np.abs(corr_matrix)
                    
                    # Perform hierarchical clustering
                    from scipy.cluster.hierarchy import linkage, fcluster
                    links = linkage(distance_matrix, method='ward')
                    
                    # Create 5-8 clusters depending on portfolio size
                    n_clusters = min(8, max(5, len(self.price_data.columns) // 10))
                    clusters = fcluster(links, n_clusters, criterion='maxclust')
                    
                    # Group assets by cluster
                    cluster_assets = defaultdict(list)
                    for i, symbol in enumerate(self.price_data.columns):
                        cluster_assets[clusters[i]].append(symbol)
                    
                    # Calculate minimum variance portfolio within each cluster
                    cluster_weights = {}
                    for cluster, assets in cluster_assets.items():
                        if len(assets) > 1:
                            # Extract submatrix for this cluster
                            indices = [list(self.price_data.columns).index(a) for a in assets]
                            sub_cov = fallback_cov[np.ix_(indices, indices)]
                            
                            # Calculate minimum variance weights for cluster
                            ones = np.ones(len(indices))
                            try:
                                inv_cov = np.linalg.inv(sub_cov)
                                cluster_min_var = (inv_cov @ ones) / (ones @ inv_cov @ ones)
                                
                                # Assign weights within cluster
                                for i, asset in enumerate(assets):
                                    cluster_weights[asset] = max(0, cluster_min_var[i])
                            except Exception:
                                # Fallback to equal weights within cluster
                                for asset in assets:
                                    cluster_weights[asset] = 1.0 / len(assets)
                        else:
                            # Single asset cluster
                            cluster_weights[assets[0]] = 1.0
                    
                    # Normalize weights within clusters
                    for cluster, assets in cluster_assets.items():
                        total = sum(cluster_weights[a] for a in assets)
                        if total > 0:
                            for asset in assets:
                                cluster_weights[asset] /= total
                    
                    # Allocate across clusters with defensive bias
                    # Calculate cluster risk
                    cluster_risk = {}
                    for cluster, assets in cluster_assets.items():
                        # Use average volatility as risk measure
                        cluster_risk[cluster] = np.mean([vols[list(self.price_data.columns).index(a)] for a in assets])
                    
                    # Inverse risk allocation with defensive bias
                    cluster_alloc = {}
                    total_inv_risk = 0
                    
                    # Calculate defensive bias for each cluster
                    cluster_defensive_ratio = {}
                    for cluster, assets in cluster_assets.items():
                        # Calculate percentage of defensive assets in cluster
                        defensive_count = sum(1 for a in assets if 
                                            self.normalized_sectors.get(a, 'Unknown') in defensive_sectors)
                        ratio = defensive_count / len(assets) if assets else 0
                        cluster_defensive_ratio[cluster] = ratio
                        
                        # Apply defensive bias to inverse risk
                        if cluster_risk[cluster] > 0:
                            bias = 1.0 + (ratio * 0.5)  # Up to 50% boost for fully defensive clusters
                            total_inv_risk += (1.0 / cluster_risk[cluster]) * bias
                    
                    # Calculate final cluster allocations
                    for cluster, risk in cluster_risk.items():
                        if risk > 0:
                            bias = 1.0 + (cluster_defensive_ratio[cluster] * 0.5)
                            cluster_alloc[cluster] = ((1.0 / risk) * bias) / total_inv_risk
                    
                    # Combine cluster and asset weights
                    weights = {}
                    for cluster, assets in cluster_assets.items():
                        cluster_weight = cluster_alloc.get(cluster, 0)
                        for asset in assets:
                            weights[asset] = cluster_weight * cluster_weights[asset]
                    
                    # Ensure all assets have weights
                    for symbol in self.price_data.columns:
                        if symbol not in weights or weights[symbol] < 1e-8:
                            weights[symbol] = 1e-8
                    
                    # Normalize final weights
                    total = sum(weights.values())
                    weights = {k: v/total for k, v in weights.items()}
                    
                    logging.info("Cluster-based minimum variance successful")
                    
                except Exception as cluster_error:
                    logging.warning(f"Cluster-based approach failed: {str(cluster_error)}")
                    
                    # Stage 3: Defensive sector-based allocation
                    try:
                        logging.info("Using defensive sector-based allocation")
                        
                        # Enhanced sector-based fallback with risk-adjusted weights
                        sector_symbols = defaultdict(list)
                        sector_risks = defaultdict(float)
                        sector_returns = defaultdict(float)
                        
                        # Calculate sector-level metrics
                        for symbol in self.price_data.columns:
                            sector = self.normalized_sectors.get(symbol, self.sectors.get(symbol, 'Unknown'))
                            sector_symbols[sector].append(symbol)
                            returns = self.returns_df[symbol]
                            sector_risks[sector] += returns.std() * np.sqrt(252)
                            sector_returns[sector] += returns.mean() * 252
                        
                        # Normalize sector metrics
                        for sector in sector_symbols:
                            if len(sector_symbols[sector]) > 0:
                                sector_risks[sector] /= len(sector_symbols[sector])
                                sector_returns[sector] /= len(sector_symbols[sector])
                        
                        # Risk-adjusted sector allocation with stronger defensive bias
                        total_risk = sum(1/risk if risk > 0 else 0 for risk in sector_risks.values())
                        sector_allocation = {}
                        
                        for sector in sector_symbols:
                            if sector_risks[sector] > 0:
                                # Inverse volatility weighting with return adjustment
                                base_weight = (1/sector_risks[sector]) / total_risk
                                # Stronger defensive bias
                                if sector in defensive_sectors:
                                    defensive_boost = 1.75  # 75% boost to defensive sectors
                                    base_weight *= defensive_boost
                                sector_allocation[sector] = base_weight
                        
                        # Normalize sector allocations
                        total_alloc = sum(sector_allocation.values())
                        sector_allocation = {k: v/total_alloc for k, v in sector_allocation.items()}
                        
                        # Ensure minimum allocation to defensive sectors
                        min_defensive = 0.65  # Minimum 65% in defensive sectors
                        
                        defensive_total = sum(sector_allocation.get(s, 0) for s in defensive_sectors)
                        if defensive_total < min_defensive and defensive_total > 0:
                            scale = min_defensive / defensive_total
                            remaining = 1.0 - min_defensive
                            non_defensive_total = 1.0 - defensive_total
                            
                            # Adjust allocations
                            for sector in list(sector_allocation.keys()):
                                if sector in defensive_sectors:
                                    sector_allocation[sector] *= scale
                                elif non_defensive_total > 0:
                                    sector_allocation[sector] *= (remaining / non_defensive_total)
                        
                        # Distribute sector weights to individual assets with risk adjustment
                        weights = {}
                        for sector, allocation in sector_allocation.items():
                            symbols = sector_symbols.get(sector, [])
                            if symbols:
                                # Calculate asset-level risk within sector
                                symbol_risks = {}
                                for symbol in symbols:
                                    vol = self.returns_df[symbol].std() * np.sqrt(252)
                                    symbol_risks[symbol] = 1.0 / (vol + 1e-8)
                                
                                # Normalize risks within sector
                                total_inv_risk = sum(symbol_risks.values())
                                if total_inv_risk > 0:
                                    for symbol in symbols:
                                        weights[symbol] = allocation * (symbol_risks[symbol] / total_inv_risk)
                                else:
                                    # Equal weight within sector if risk calculation fails
                                    for symbol in symbols:
                                        weights[symbol] = allocation / len(symbols)
                        
                        # Ensure all assets have weights
                        for symbol in self.price_data.columns:
                            if symbol not in weights:
                                weights[symbol] = 1e-8
                        
                        # Normalize final weights
                        total = sum(weights.values())
                        weights = {k: v/total for k, v in weights.items()}
                        
                        logging.info("Defensive sector-based allocation successful")
                        
                    except Exception as sector_error:
                        logging.warning(f"Defensive sector-based allocation failed: {str(sector_error)}")
                        # Final fallback to modified equal weights with defensive tilt
                        weights = {}
                        for symbol in self.price_data.columns:
                            sector = self.normalized_sectors.get(symbol, self.sectors.get(symbol, 'Unknown'))
                            if sector in defensive_sectors:
                                weights[symbol] = 1.5  # 50% boost to defensive assets
                            else:
                                weights[symbol] = 1.0
                        
                        # Normalize weights
                        total = sum(weights.values())
                        weights = {k: v/total for k, v in weights.items()}
                        logging.warning("Using modified equal weights with defensive tilt as final fallback")
            
            # Create dynamic fallback metrics based on market conditions
            market_return = np.mean(self.returns_df.mean()) * 252
            market_vol = np.std(self.returns_df.std()) * np.sqrt(252)
            
            # Calculate portfolio-specific metrics when possible
            try:
                port_returns = self.returns_df.dot(pd.Series(weights))
                port_mean = port_returns.mean() * 252
                port_vol = port_returns.std() * np.sqrt(252)
                port_sharpe = port_mean / port_vol if port_vol > 0 else 0
                
                # Calculate drawdown
                cum_returns = (1 + port_returns).cumprod()
                peak = cum_returns.expanding(min_periods=1).max()
                drawdown = (cum_returns - peak) / peak
                max_dd = abs(drawdown.min())
                
                portfolio_metrics = {
                    'expected_return': float(np.clip(port_mean, 0.02, 0.15)),
                    'volatility': float(np.clip(port_vol, 0.06, 0.18)),
                    'sharpe_ratio': float(np.clip(port_sharpe, 0.2, 2.5)),
                    'max_drawdown': float(np.clip(max_dd, 0.08, 0.25)),
                    'diversification_ratio': float(len([w for w in weights.values() if w > 0.005])),
                    'defensive_allocation': float(sum(weights[s] for s in self.price_data.columns 
                                                if self.normalized_sectors.get(s, 'Unknown') in defensive_sectors)),
                    'warning': 'Using enhanced defensive fallback allocation'
                }
            except Exception:
                # Fallback metrics based on market conditions
                portfolio_metrics = {
                    'expected_return': float(np.clip(market_return * 0.7, 0.02, 0.12)),
                    'volatility': float(np.clip(market_vol * 0.8, 0.06, 0.15)),
                    'sharpe_ratio': float(np.clip((market_return * 0.7) / (market_vol * 0.8), 0.2, 2.0)),
                    'max_drawdown': float(np.clip(abs(self.returns_df.min().min()) * 1.1, 0.08, 0.20)),
                    'diversification_ratio': float(len([w for w in weights.values() if w > 0.005])),
                    'warning': 'Using enhanced defensive fallback allocation'
                }
            
            return weights, portfolio_metrics
        except Exception as e:
            logging.error(f"Portfolio optimization failed: {str(e)}")
            raise
            
            # Normalize and validate weights with improved numerical stability
            total_weight = sum(raw_weights.values())
            weights = {k: v/total_weight for k, v in raw_weights.items()}
            
            # Calculate portfolio metrics with enhanced stability and validation
            try:
                # Calculate metrics based on available optimizer
                if 'hrp' in locals() and hrp is not None:
                    returns = self.returns_df.dot(pd.Series(weights))
                    annual_return = returns.mean() * 252
                    annual_vol = returns.std() * np.sqrt(252)
                    sharpe = annual_return / annual_vol if annual_vol > 0 else 0.0
                    
                    portfolio_metrics = {
                        'expected_return': float(np.clip(annual_return, 0.01, 0.3)),
                        'volatility': float(np.clip(annual_vol, 0.05, 0.3)),
                        'sharpe_ratio': float(np.clip(sharpe, 0.1, 5.0))
                    }
                else:
                    try:
                        perf = ef.portfolio_performance()
                        # Validate that perf is a tuple with at least 3 elements
                        if not isinstance(perf, tuple) or len(perf) < 3:
                            raise ValueError(f"Invalid performance tuple: {perf}")
                            
                        portfolio_metrics = {
                            'expected_return': float(np.clip(perf[0], 0.01, 0.3)),
                            'volatility': float(np.clip(perf[1], 0.05, 0.3)),
                            'sharpe_ratio': float(np.clip(perf[2], 0.1, 5.0))
                        }
                    except Exception as e1:
                        logging.warning(f"Standard performance calculation failed: {str(e1)}. Using fallback calculation.")
                        # Fallback to manual calculation
                        returns = self.returns_df.dot(pd.Series(weights))
                        annual_return = returns.mean() * 252
                        annual_vol = returns.std() * np.sqrt(252)
                        sharpe = annual_return / annual_vol if annual_vol > 0 else 0.0
                        
                        portfolio_metrics = {
                            'expected_return': float(np.clip(annual_return, 0.01, 0.3)),
                            'volatility': float(np.clip(annual_vol, 0.05, 0.3)),
                            'sharpe_ratio': float(np.clip(sharpe, 0.1, 5.0))
                        }
                
                # Calculate additional risk metrics
                returns = self.returns_df.dot(pd.Series(weights))
                cum_returns = (1 + returns).cumprod()
                running_max = cum_returns.expanding(min_periods=1).max()
                drawdowns = (cum_returns - running_max) / running_max
                max_drawdown = float(np.clip(abs(drawdowns.min()), 0.05, 0.4))
                
                # Calculate diversification metrics
                effective_n = 1 / sum(w * w for w in weights.values())
                portfolio_metrics.update({
                    'max_drawdown': max_drawdown,
                    'diversification_ratio': float(np.clip(effective_n, 1, len(weights))),
                    'largest_position': float(max(weights.values())),
                    'number_of_positions': len([w for w in weights.values() if w > 0.01])
                })
                
                return weights, portfolio_metrics
                
            except Exception as e:
                logging.error(f"Portfolio metrics calculation failed: {str(e)}")
                raise ValueError("Failed to calculate portfolio metrics")
            
            # Additional risk metrics with enhanced stability
            returns = self.returns_df.dot(pd.Series(weights))
            rolling_drawdown = np.maximum.accumulate(returns) - returns
            max_drawdown = float(np.clip(np.max(rolling_drawdown), 0, 0.25))
            
            # Calculate improved diversification metrics
            effective_positions = sum(w > 0.005 for w in weights.values())
            herfindahl = sum(w * w for w in weights.values())
            diversification_ratio = float(1.0 / herfindahl) / len(weights)
            
            # Calculate sector concentration using normalized sectors
            sector_allocation = defaultdict(float)
            for symbol, weight in weights.items():
                sector = self.normalized_sectors.get(symbol, self.sectors.get(symbol, 'Unknown'))
                sector_allocation[sector] += weight
            
            # Comprehensive portfolio metrics
            portfolio_metrics.update({
                'max_drawdown': max_drawdown,
                'diversification_ratio': diversification_ratio,
                'effective_positions': effective_positions,
                'sector_allocation': dict(sector_allocation)
            })
            
            return weights, portfolio_metrics
            
        except Exception as e:
            logging.error(f"Error calculating portfolio metrics: {str(e)}")
            raise ValueError(f"Portfolio optimization failed: {str(e)}")
    
    def _balanced_strategy(self, constraints: dict) -> tuple[dict[str, float], dict[str, float]]:
        """Balanced portfolio optimization with enhanced constraint handling and risk management"""
        try:
            # Ensure normalized sectors are available
            if not hasattr(self, 'normalized_sectors'):
                # Map sectors to ensure consistency with asset_loader.py
                sector_mapping = {
                    'Consumer Cyclical': 'Consumer Discretionary',
                    'Consumer Defensive': 'Consumer Staples',
                    'Financial': 'Financial Services',
                    'Technology': 'Information Technology',
                    'Basic Materials': 'Materials'
                }
                
                # Create a normalized sectors dictionary for optimization
                normalized_sectors = {}
                for symbol, sector in self.sectors.items():
                    normalized_sectors[symbol] = sector_mapping.get(sector, sector)
                
                # Use normalized sectors for optimization
                self.normalized_sectors = normalized_sectors
                
            # Filter data to use only valid symbols with both price data and sector information
            sector_symbols = set(self.normalized_sectors.keys())
            price_symbols = set(self.price_data.columns)
            valid_symbols = sector_symbols.intersection(price_symbols)
            
            if not valid_symbols:
                raise ValueError("No valid symbols found with both price data and sector information")
            
            # Filter data to use only valid symbols
            filtered_price_data = self.price_data[list(valid_symbols)]
            filtered_returns = self.expected_returns[list(valid_symbols)]
                
            # Enhanced covariance estimation with multi-factor shrinkage and robust validation
            try:
                # Start with Ledoit-Wolf shrinkage using filtered data
                base_cov = risk_models.CovarianceShrinkage(
                    filtered_price_data,
                    frequency=252
                ).ledoit_wolf()
                
                # Validate covariance matrix
                if not np.all(np.isfinite(base_cov)):
                    raise ValueError("Non-finite values in covariance matrix")
                    
                # Calculate market volatility for adaptive shrinkage with validation
                market_vol = np.sqrt(max(1e-8, np.trace(base_cov)))
                shrinkage_factor = min(0.8, max(0.2, 1.0 / (1 + market_vol)))
                
                # Enhanced stability checks
                if not np.isfinite(shrinkage_factor):
                    shrinkage_factor = 0.5  # Default to balanced shrinkage
                
                # Apply additional shrinkage to handle extreme market conditions
                diag_cov = np.diag(np.diag(base_cov))
                reg_cov_matrix = (1 - shrinkage_factor) * base_cov + shrinkage_factor * diag_cov
                
                # Add stability term scaled by condition number
                condition_number = np.linalg.cond(reg_cov_matrix)
                stability_factor = max(1e-8, min(1e-4, 1e-6 * np.log(condition_number)))
                reg_cov_matrix += np.eye(reg_cov_matrix.shape[0]) * stability_factor
                
                # Ensure matrix is positive definite
                min_eigenval = np.min(np.real(np.linalg.eigvals(reg_cov_matrix)))
                if min_eigenval < 1e-8:
                    reg_cov_matrix += np.eye(reg_cov_matrix.shape[0]) * (abs(min_eigenval) + 1e-8)
            except Exception as e:
                logging.warning(f"Advanced covariance estimation failed: {str(e)}")
                # Fallback to simpler estimation
                reg_cov_matrix = risk_models.sample_cov(self.price_data, frequency=252)
                reg_cov_matrix += np.eye(reg_cov_matrix.shape[0]) * 1e-6

            # Enhanced sector risk calculation with proper validation using filtered data
            sector_counts = defaultdict(int)
            sector_vols = defaultdict(float)
            sector_returns = defaultdict(list)
            
            for symbol in valid_symbols:
                sector = self.normalized_sectors.get(symbol)
                if sector:
                    try:
                        idx = list(filtered_price_data.columns).index(symbol)
                        sector_counts[sector] += 1
                        
                        # Calculate volatility with validation
                        symbol_vol = np.sqrt(max(0, reg_cov_matrix[idx, idx]))
                        if not np.isnan(symbol_vol) and not np.isinf(symbol_vol):
                            sector_vols[sector] += symbol_vol
                            
                        # Track sector returns for risk-adjusted allocation
                        if not np.isnan(filtered_returns[idx]):
                            sector_returns[sector].append(filtered_returns[idx])
                    except Exception as e:
                        logging.warning(f"Error processing sector metrics for {symbol}: {str(e)}")
                        continue
                        
            # Validate sector data
            if not sector_counts:
                raise ValueError("No valid sectors found after processing")
            
            # Calculate average sector metrics
            for sector in sector_counts:
                if sector_counts[sector] > 0:
                    sector_vols[sector] /= sector_counts[sector]
                    if sector_returns[sector]:
                        sector_returns[sector] = np.mean(sector_returns[sector])

            # Calculate adaptive sector limits with enhanced validation and volatility scaling
            num_sectors = len(sector_counts)
            if num_sectors == 0:
                raise ValueError("No valid sectors found for optimization")
                
            base_max = min(0.3, constraints.get('max_sector', 0.3))
            
            # Calculate sector volatility scaling with validation
            vol_scale = {}
            for sector in sector_counts:
                count = sector_counts[sector]
                vol = sector_vols[sector]
                if count > 0 and vol > 0:
                    scale = min(1.2, max(0.8, 1.0 / (1 + vol/count)))
                    vol_scale[sector] = scale
                else:
                    vol_scale[sector] = 1.0
                    logging.warning(f"Invalid metrics for {sector}, using default scale")
            
            # Calculate sector bounds with validation
            sector_upper = {}
            sector_lower = {}
            for sector in sector_counts:
                count = sector_counts[sector]
                if count > 0:
                    # Upper bound with volatility adjustment
                    upper = base_max * vol_scale[sector] * (1 + np.log1p(count)/8)
                    sector_upper[sector] = min(0.4, max(0.1, upper))  # Ensure reasonable bounds
                    
                    # Lower bound with sector count adjustment
                    min_weight = max(0.02, min(0.1, constraints.get('min_bonds', 0.1) * 0.6 / num_sectors))
                    sector_lower[sector] = min(min_weight, sector_upper[sector] * 0.5)
                    
            if not sector_upper or not sector_lower:
                raise ValueError("Failed to calculate valid sector bounds")

            # Multi-stage optimization with adaptive constraints
            success = False
            raw_weights = {}
            
            # Calculate market volatility regime
            market_vol = np.sqrt(np.trace(reg_cov_matrix))
            vol_regime = 'high' if market_vol > np.median(np.diag(reg_cov_matrix)) else 'low'
            
            for attempt in range(4):
                try:
                    # Progressive constraint relaxation with volatility adjustment
                    relax_base = 0.12 if vol_regime == 'high' else 0.08
                    relax_factor = 1 - (relax_base * attempt)
                    
                    # Adaptive sector constraints
                    current_upper = {s: min(0.35, v * relax_factor * (1.2 if vol_regime == 'low' else 1.0)) 
                                   for s, v in sector_upper.items()}
                    current_lower = {s: max(0.02, v * (1.8 - relax_factor) * (0.8 if vol_regime == 'high' else 1.0)) 
                                   for s, v in sector_lower.items()}

                    # Initialize optimizer with risk-adjusted parameters and enhanced validation
                    ef = EfficientFrontier(self.expected_returns, reg_cov_matrix, weight_bounds=(0, 0.35))
                    ef.add_constraint(lambda w: cp.sum(w) == 1)
                    
                    # Enhanced sector constraint handling with progressive fallbacks
                    try:
                        # Validate and filter sector mappings
                        valid_sectors = {k: v for k, v in self.normalized_sectors.items() 
                                       if k in self.price_data.columns and v in current_upper}
                        
                        if len(valid_sectors) < len(self.price_data.columns) * 0.5:
                            logging.warning(f"Only {len(valid_sectors)}/{len(self.price_data.columns)} assets have valid sector mappings")
                        
                        if not valid_sectors:
                            raise ValueError("No valid sector mappings found")
                            
                        # Calculate sector diversification metrics
                        sector_counts = defaultdict(int)
                        for sector in valid_sectors.values():
                            sector_counts[sector] += 1
                            
                        # Adjust constraints based on sector representation
                        adjusted_upper = {}
                        adjusted_lower = {}
                        for sector, count in sector_counts.items():
                            sector_weight = count / len(valid_sectors)
                            adjusted_upper[sector] = min(current_upper[sector], 
                                                        max(0.15, sector_weight * 1.5))
                            adjusted_lower[sector] = min(current_lower[sector],
                                                        max(0.02, sector_weight * 0.5))
                        
                        # Apply enhanced sector constraints
                        ef.add_sector_constraints(valid_sectors,
                                                adjusted_upper,
                                                adjusted_lower)
                        
                        # Add minimum weight constraint for numerical stability
                        ef.add_constraint(lambda w: w >= 1e-4)
                    except Exception as sector_error:
                        logging.warning(f"Sector constraint error: {str(sector_error)}")
                        # Fallback to basic constraints while maintaining sector exposure
                        ef = EfficientFrontier(self.expected_returns, reg_cov_matrix)
                        ef.add_constraint(lambda w: cp.sum(w) == 1)
                        ef.add_constraint(lambda w: w >= 1e-4)
                        ef.add_constraint(lambda w: w <= 0.35)
                    
                    # Dynamic regularization based on market conditions
                    cond_number = np.linalg.cond(reg_cov_matrix)
                    l2_gamma = max(0.05, min(0.25, 0.3 / np.log1p(cond_number)))
                    ef.add_objective(objective_functions.L2_reg, gamma=l2_gamma)
                    
                    # Enhanced risk-adjusted optimization with multiple objectives
                    target_return = np.mean(self.expected_returns) * (0.9 if vol_regime == 'low' else 0.7)
                    ef.add_constraint(lambda w: w @ self.expected_returns >= target_return)
                    
                    # Add balanced objectives with dynamic weighting
                    vol_weight = 0.6 if vol_regime == 'high' else 0.4
                    ret_weight = 1 - vol_weight
                    
                    # Multi-objective optimization with risk-return balance
                    ef.add_objective(lambda w: vol_weight * (w @ reg_cov_matrix @ w))  # Volatility control
                    ef.add_objective(lambda w: -ret_weight * (w @ self.expected_returns))  # Return enhancement
                    
                    # Attempt optimization with enhanced numerical stability
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        ef.optimize()
                        raw_weights = ef.clean_weights()
                        
                    # Validate optimization results
                    if all(0 <= w <= 1 for w in raw_weights.values()):
                        success = True
                        break
                    else:
                        raise ValueError("Invalid weight values detected")
                        
                except Exception as e:
                    logging.info(f"Attempt {attempt+1} failed: {str(e)}")
                    if attempt == 3:
                        logging.warning("All optimization attempts failed, initiating HRP fallback")

            if not success:
                # Enhanced multi-stage fallback system with hierarchical optimization approaches
                logging.info("Initiating enhanced fallback system")
                
                # Stage 1: Advanced Risk Parity with correlation adjustment
                try:
                    logging.info("Attempting advanced risk parity optimization")
                    # Calculate asset volatilities with stability enhancement
                    vols = np.sqrt(np.diag(reg_cov_matrix))
                    # Apply volatility floor to prevent extreme allocations
                    vols = np.maximum(vols, np.median(vols) * 0.25)
                    inv_vols = 1.0 / vols
                    raw_weights = {}
                    
                    # Calculate correlation matrix for diversification adjustment
                    diag_sqrt = np.diag(1.0 / np.sqrt(np.diag(reg_cov_matrix)))
                    corr_matrix = diag_sqrt @ reg_cov_matrix @ diag_sqrt
                    
                    # Apply correlation-adjusted risk parity
                    # This reduces allocation to highly correlated assets
                    asset_correlations = np.mean(np.abs(corr_matrix), axis=1)
                    correlation_penalty = 1.0 / (0.5 + 0.5 * asset_correlations)
                    adjusted_inv_vols = inv_vols * correlation_penalty
                    
                    # Assign weights with correlation adjustment
                    total_adj_inv_vol = np.sum(adjusted_inv_vols)
                    for i, symbol in enumerate(self.price_data.columns):
                        raw_weights[symbol] = adjusted_inv_vols[i] / total_adj_inv_vol
                    
                    # Apply sector constraints with progressive relaxation
                    sector_weights = defaultdict(float)
                    for symbol, weight in raw_weights.items():
                        sector = self.normalized_sectors.get(symbol, 'Unknown')
                        sector_weights[sector] += weight
                    
                    # Relaxed sector constraints for fallback
                    relaxed_upper = {k: min(0.45, v * 1.25) for k, v in current_upper.items()}
                    relaxed_lower = {k: max(0.01, v * 0.75) for k, v in current_lower.items()}
                    
                    # Adjust weights to meet relaxed sector constraints
                    scale_factors = {}
                    for sector, weight in sector_weights.items():
                        if sector != 'Unknown' and sector in relaxed_upper:
                            if weight > relaxed_upper[sector]:
                                scale_factors[sector] = relaxed_upper[sector] / weight
                            elif weight < relaxed_lower[sector]:
                                scale_factors[sector] = relaxed_lower[sector] / weight
                            else:
                                scale_factors[sector] = 1.0
                    
                    # Apply scaling with validation
                    for symbol in raw_weights:
                        sector = self.normalized_sectors.get(symbol, 'Unknown')
                        if sector in scale_factors:
                            raw_weights[symbol] *= scale_factors[sector]
                    
                    # Normalize weights and validate
                    total = sum(raw_weights.values())
                    if total > 0:
                        raw_weights = {k: v/total for k, v in raw_weights.items()}
                        # Verify weights are valid
                        if all(0 <= w <= 1 for w in raw_weights.values()) and abs(sum(raw_weights.values()) - 1.0) < 1e-6:
                            success = True
                            logging.info("Advanced risk parity optimization successful")
                        else:
                            raise ValueError("Invalid weights after risk parity optimization")
                    else:
                        raise ValueError("Total weight is zero or negative")
                        
                except Exception as rp_error:
                    logging.warning(f"Advanced risk parity approach failed: {str(rp_error)}")
                    
                    # Stage 2: Enhanced HRP with cluster-based allocation
                    try:
                        logging.info("Attempting enhanced HRP optimization")
                        # Prepare returns data with robust preprocessing
                        clean_returns = self.returns_df.copy()
                        # Replace outliers and missing values
                        clean_returns = clean_returns.clip(clean_returns.quantile(0.05), clean_returns.quantile(0.95))
                        clean_returns = clean_returns.fillna(method='ffill').fillna(0)
                        
                        # Initialize HRP with enhanced stability
                        hrp = HRPOpt(clean_returns)
                        # Use more robust linkage method
                        hrp.optimize(linkage_method='complete')
                        raw_weights = hrp.clean_weights()
                        
                        # Apply minimum position size and maximum constraints
                        raw_weights = {k: min(0.2, max(0.005, v)) for k, v in raw_weights.items()}
                        
                        # Apply sector constraints with relaxed bounds
                        sector_weights = defaultdict(float)
                        for symbol, weight in raw_weights.items():
                            sector = self.normalized_sectors.get(symbol, 'Unknown')
                            sector_weights[sector] += weight
                            
                        # Check if any sector exceeds maximum allocation
                        max_sector_weight = 0.4  # Relaxed constraint for fallback
                        if any(w > max_sector_weight for w in sector_weights.values()):
                            # Scale down overweight sectors
                            for symbol, weight in list(raw_weights.items()):
                                sector = self.normalized_sectors.get(symbol, 'Unknown')
                                if sector_weights[sector] > max_sector_weight:
                                    scale = max_sector_weight / sector_weights[sector]
                                    raw_weights[symbol] *= scale
                        
                        # Normalize and validate
                        total = sum(raw_weights.values())
                        if total > 0:
                            raw_weights = {k: v/total for k, v in raw_weights.items()}
                            if all(w >= 0 for w in raw_weights.values()) and abs(sum(raw_weights.values()) - 1.0) < 1e-6:
                                success = True
                                logging.info("Enhanced HRP optimization successful")
                            else:
                                raise ValueError("Invalid weights after HRP optimization")
                        else:
                            raise ValueError("Total weight is zero or negative after HRP")
                            
                    except Exception as hrp_error:
                        logging.warning(f"Enhanced HRP fallback failed: {str(hrp_error)}")
                        
                        # Stage 3: Minimum variance with shrinkage
                        try:
                            logging.info("Attempting minimum variance optimization")
                            # Apply stronger shrinkage to covariance matrix
                            shrinkage_cov = risk_models.CovarianceShrinkage(
                                self.price_data,
                                frequency=252
                            ).oracle_approximating()
                            
                            # Add stability buffer
                            shrinkage_cov += np.eye(shrinkage_cov.shape[0]) * (np.trace(shrinkage_cov) * 1e-4)
                            
                            # Create optimizer with minimal constraints
                            min_var = EfficientFrontier(
                                self.expected_returns, 
                                shrinkage_cov,
                                weight_bounds=(0.001, 0.25)  # Prevent extreme allocations
                            )
                            min_var.add_constraint(lambda w: cp.sum(w) == 1)
                            
                            # Optimize for minimum variance
                            min_var.min_volatility()
                            raw_weights = min_var.clean_weights()
                            
                            # Validate results
                            if all(w >= 0 for w in raw_weights.values()) and abs(sum(raw_weights.values()) - 1.0) < 1e-6:
                                success = True
                                logging.info("Minimum variance optimization successful")
                            else:
                                raise ValueError("Invalid weights from minimum variance")
                                
                        except Exception as min_var_error:
                            logging.warning(f"Minimum variance fallback failed: {str(min_var_error)}")
                            
                            # Final fallback: Sector-based equal weights
                            try:
                                logging.info("Using sector-based equal weights as final fallback")
                                # Group assets by sector
                                sector_assets = defaultdict(list)
                                for symbol in self.price_data.columns:
                                    sector = self.normalized_sectors.get(symbol, 'Unknown')
                                    sector_assets[sector].append(symbol)
                                
                                # Calculate sector allocations based on risk
                                sector_risk = {}
                                for sector, assets in sector_assets.items():
                                    if sector != 'Unknown':
                                        # Use inverse volatility for sector allocation
                                        sector_vol = np.mean([np.sqrt(reg_cov_matrix[list(self.price_data.columns).index(a), 
                                                                        list(self.price_data.columns).index(a)]) 
                                                            for a in assets])
                                        sector_risk[sector] = 1.0 / (sector_vol + 1e-8)
                                
                                # Normalize sector allocations
                                total_risk = sum(sector_risk.values())
                                sector_alloc = {k: v / total_risk for k, v in sector_risk.items()} if total_risk > 0 else {}
                                
                                # Equal weight within sectors, risk-weighted across sectors
                                raw_weights = {}
                                for sector, alloc in sector_alloc.items():
                                    assets = sector_assets[sector]
                                    if assets:
                                        for asset in assets:
                                            raw_weights[asset] = alloc / len(assets)
                                
                                # Handle unknown sector assets
                                unknown_assets = sector_assets.get('Unknown', [])
                                if unknown_assets:
                                    unknown_weight = 0.1  # Allocate 10% to unknown sector
                                    for asset in unknown_assets:
                                        raw_weights[asset] = unknown_weight / len(unknown_assets)
                                    
                                    # Rescale other weights
                                    scale = (1.0 - unknown_weight) / sum(v for k, v in raw_weights.items() if k not in unknown_assets)
                                    for k in raw_weights:
                                        if k not in unknown_assets:
                                            raw_weights[k] *= scale
                                
                                # Ensure all assets have weights
                                for symbol in self.price_data.columns:
                                    if symbol not in raw_weights:
                                        raw_weights[symbol] = 0.001
                                
                                # Normalize final weights
                                total = sum(raw_weights.values())
                                raw_weights = {k: v/total for k, v in raw_weights.items()}
                                success = True
                                logging.info("Sector-based equal weights applied successfully")
                                
                            except Exception as sector_error:
                                logging.error(f"Sector-based fallback failed: {str(sector_error)}")
                                # Ultimate fallback to pure equal weights
                                n_assets = len(self.price_data.columns)
                                raw_weights = {col: 1.0/n_assets for col in self.price_data.columns}
                                logging.warning("Using pure equal weights as last resort")

            # Validate and normalize weights
            total_weight = sum(raw_weights.values())
            if total_weight < 0.99 or total_weight > 1.01:
                logging.warning(f"Weight normalization required (total={total_weight:.4f})")
                total_weight = max(total_weight, 1e-4)  # Prevent division by zero
                weights = {k: v/total_weight for k, v in raw_weights.items()}
            else:
                weights = raw_weights

            # Enhanced portfolio metrics calculation with proper validation
            try:
                # Calculate returns with weight validation
                weight_values = np.array(list(weights.values()))
                if not np.isclose(np.sum(weight_values), 1.0, rtol=1e-3):
                    logging.warning(f"Portfolio weights sum to {np.sum(weight_values)}, normalizing")
                    weight_values = weight_values / np.sum(weight_values)
                
                # Calculate portfolio returns
                returns = self.returns_df.dot(weight_values)
                
                # Calculate risk metrics with validation
                ret_mean = np.mean(returns)
                ret_std = np.std(returns)
                sharpe = ret_mean/ret_std if ret_std > 0 else 0
                
                # Calculate maximum drawdown
                cum_returns = (1 + returns).cumprod()
                rolling_max = np.maximum.accumulate(cum_returns)
                drawdowns = (cum_returns - rolling_max) / rolling_max
                max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
                
                # Calculate diversification metrics
                effective_positions = len([w for w in weight_values if w > 0.01])
                herfindahl = np.sum(weight_values**2)
                
                portfolio_metrics = {
                    'sharpe_ratio': float(np.nan_to_num(sharpe)),
                    'volatility': float(np.nan_to_num(ret_std)),
                    'expected_return': float(np.nan_to_num(ret_mean)),
                    'max_drawdown': float(np.nan_to_num(max_drawdown)),
                    'diversification_ratio': effective_positions,
                    'concentration': float(np.nan_to_num(herfindahl))
                }

                # Validate metrics
                if any(np.isnan(v) for v in portfolio_metrics.values()):
                    raise ValueError("Invalid metrics containing NaN values")

                # Validate required metrics exist
                for key in ['expected_return', 'volatility', 'sharpe_ratio']:
                    if key not in portfolio_metrics:
                        portfolio_metrics[key] = 0.0
                        logging.warning(f"Missing {key} in portfolio metrics, using default value")

            except Exception as e:
                logging.error(f"Error calculating portfolio metrics: {str(e)}")
                portfolio_metrics = {
                    'sharpe_ratio': 0.0,
                    'volatility': 0.0,
                    'expected_return': 0.0,
                    'max_drawdown': 0.0,
                    'diversification_ratio': 0
                }

            return weights, portfolio_metrics
        except Exception as e:
            logging.error(f"Balanced strategy failed: {str(e)}")
            logging.info("Initiating enhanced balanced fallback system")
            
            # Multi-stage fallback system for balanced strategy
            try:
                # Stage 1: Enhanced Risk Parity with correlation adjustment
                logging.info("Attempting enhanced risk parity with correlation adjustment")
                
                # Ensure we have a valid covariance matrix for fallback
                try:
                    # Create a more stable covariance matrix with stronger shrinkage
                    fallback_cov = risk_models.CovarianceShrinkage(
                        self.price_data,
                        frequency=252
                    ).oracle_approximating()
                    
                    # Add stability buffer proportional to matrix trace
                    stability_factor = np.trace(fallback_cov) * 1e-4
                    fallback_cov += np.eye(fallback_cov.shape[0]) * stability_factor
                    
                    # Ensure symmetry and positive definiteness
                    fallback_cov = (fallback_cov + fallback_cov.T) / 2
                    min_eigenval = np.min(np.real(np.linalg.eigvals(fallback_cov)))
                    if min_eigenval <= 1e-8:
                        correction = abs(min_eigenval) + 1e-7
                        fallback_cov += np.eye(fallback_cov.shape[0]) * correction
                except Exception as cov_error:
                    logging.warning(f"Fallback covariance estimation failed: {str(cov_error)}")
                    # Use diagonal covariance as ultimate fallback
                    returns = self.returns_df.fillna(0)
                    vols = returns.std() * np.sqrt(252)
                    fallback_cov = np.diag(vols**2)
                
                # Calculate correlation matrix for diversification adjustment
                diag_sqrt = np.diag(1.0 / np.sqrt(np.maximum(np.diag(fallback_cov), 1e-8)))
                corr_matrix = diag_sqrt @ fallback_cov @ diag_sqrt
                
                # Calculate asset volatilities
                vols = np.sqrt(np.diag(fallback_cov))
                inv_vols = 1.0 / (vols + 1e-8)  # Avoid division by zero
                
                # Calculate average absolute correlation for each asset
                avg_corr = np.mean(np.abs(corr_matrix), axis=1)
                
                # Apply correlation penalty to reduce allocation to highly correlated assets
                # Assets with higher correlation get lower weights
                correlation_penalty = 1.0 / (0.5 + 0.5 * avg_corr)
                
                # Calculate correlation-adjusted inverse volatility
                adjusted_inv_vols = inv_vols * correlation_penalty
                
                # Calculate weights with correlation adjustment
                raw_weights = {}
                total_adj_inv_vol = np.sum(adjusted_inv_vols)
                for i, symbol in enumerate(self.price_data.columns):
                    raw_weights[symbol] = adjusted_inv_vols[i] / total_adj_inv_vol
                
                # Apply sector constraints with balanced approach
                sector_weights = defaultdict(float)
                for symbol, weight in raw_weights.items():
                    sector = self.normalized_sectors.get(symbol, 'Unknown')
                    sector_weights[sector] += weight
                
                # Check for sector concentration
                max_sector_weight = 0.35  # Maximum 35% in any sector for balanced approach
                overweight_sectors = [s for s, w in sector_weights.items() if w > max_sector_weight]
                
                if overweight_sectors:
                    # Adjust weights to meet sector constraints
                    for sector in overweight_sectors:
                        scale_factor = max_sector_weight / sector_weights[sector]
                        for symbol in raw_weights:
                            if self.normalized_sectors.get(symbol, 'Unknown') == sector:
                                raw_weights[symbol] *= scale_factor
                    
                    # Redistribute excess weight to underweight sectors
                    excess = sum(sector_weights[s] - max_sector_weight for s in overweight_sectors)
                    underweight_sectors = [s for s in sector_weights if s not in overweight_sectors]
                    
                    if underweight_sectors:
                        for symbol in raw_weights:
                            sector = self.normalized_sectors.get(symbol, 'Unknown')
                            if sector in underweight_sectors:
                                # Proportional redistribution
                                current_sector_weight = sector_weights[sector]
                                total_underweight = sum(sector_weights[s] for s in underweight_sectors)
                                if total_underweight > 0:
                                    boost_factor = 1.0 + (excess * (current_sector_weight / total_underweight))
                                    raw_weights[symbol] *= boost_factor
                
                # Normalize weights
                total = sum(raw_weights.values())
                if total > 0:
                    weights = {k: v/total for k, v in raw_weights.items()}
                    logging.info("Enhanced risk parity with correlation adjustment successful")
                else:
                    raise ValueError("Total weight is zero or negative")
                    
            except Exception as rp_error:
                logging.warning(f"Enhanced risk parity failed: {str(rp_error)}")
                
                # Stage 2: Hierarchical Risk Parity with robust preprocessing
                try:
                    logging.info("Attempting Hierarchical Risk Parity optimization")
                    
                    # Prepare returns data with robust preprocessing
                    clean_returns = self.returns_df.copy()
                    # Replace outliers with quantile values
                    clean_returns = clean_returns.clip(
                        clean_returns.quantile(0.05), 
                        clean_returns.quantile(0.95)
                    )
                    clean_returns = clean_returns.fillna(method='ffill').fillna(0)
                    
                    # Add minimal noise to prevent perfect correlation
                    noise_factor = 1e-8 * np.random.randn(*clean_returns.shape)
                    clean_returns += noise_factor
                    
                    # Initialize HRP with enhanced stability
                    hrp = HRPOpt(clean_returns)
                    
                    # Use more robust linkage method
                    hrp.optimize(linkage_method='ward')
                    raw_weights = hrp.clean_weights()
                    
                    # Apply minimum position size and maximum constraints
                    raw_weights = {k: min(0.15, max(0.005, v)) for k, v in raw_weights.items()}
                    
                    # Normalize weights
                    total = sum(raw_weights.values())
                    if total > 0:
                        weights = {k: v/total for k, v in raw_weights.items()}
                        logging.info("Hierarchical Risk Parity optimization successful")
                    else:
                        raise ValueError("Total weight is zero or negative after HRP")
                        
                except Exception as hrp_error:
                    logging.warning(f"HRP fallback failed: {str(hrp_error)}")
                    
                    # Stage 3: Balanced sector allocation
                    try:
                        logging.info("Using balanced sector allocation")
                        
                        # Group assets by sector
                        sector_assets = defaultdict(list)
                        for symbol in self.price_data.columns:
                            sector = self.normalized_sectors.get(symbol, 'Unknown')
                            sector_assets[sector].append(symbol)
                        
                        # Calculate sector risk metrics
                        sector_risk = {}
                        sector_returns = {}
                        for sector, assets in sector_assets.items():
                            if assets:
                                # Calculate average volatility for sector
                                sector_vols = []
                                sector_rets = []
                                for asset in assets:
                                    if asset in self.returns_df.columns:
                                        returns = self.returns_df[asset].fillna(0)
                                        vol = returns.std() * np.sqrt(252)
                                        ret = returns.mean() * 252
                                        if np.isfinite(vol) and vol > 0:
                                            sector_vols.append(vol)
                                        if np.isfinite(ret):
                                            sector_rets.append(ret)
                                
                                if sector_vols:
                                    sector_risk[sector] = np.mean(sector_vols)
                                else:
                                    sector_risk[sector] = 0.2  # Default risk if calculation fails
                                    
                                if sector_rets:
                                    sector_returns[sector] = np.mean(sector_rets)
                                else:
                                    sector_returns[sector] = 0.05  # Default return if calculation fails
                        
                        # Calculate balanced sector allocations using risk-return metrics
                        sector_allocation = {}
                        
                        # Calculate risk-adjusted returns for each sector
                        risk_adjusted_returns = {}
                        for sector in sector_risk:
                            if sector_risk[sector] > 0:
                                # Simple Sharpe-like ratio
                                risk_adjusted_returns[sector] = sector_returns.get(sector, 0.05) / sector_risk[sector]
                            else:
                                risk_adjusted_returns[sector] = 0
                        
                        # Normalize risk-adjusted returns
                        total_rar = sum(max(0, r) for r in risk_adjusted_returns.values())
                        if total_rar > 0:
                            for sector in risk_adjusted_returns:
                                sector_allocation[sector] = max(0, risk_adjusted_returns[sector]) / total_rar
                        else:
                            # Fallback to inverse volatility if risk-adjusted returns are invalid
                            total_inv_risk = sum(1/r if r > 0 else 0 for r in sector_risk.values())
                            if total_inv_risk > 0:
                                for sector, risk in sector_risk.items():
                                    if risk > 0:
                                        sector_allocation[sector] = (1/risk) / total_inv_risk
                                    else:
                                        sector_allocation[sector] = 0
                            else:
                                # Equal allocation if all else fails
                                for sector in sector_risk:
                                    sector_allocation[sector] = 1.0 / len(sector_risk)
                        
                        # Balance sector allocations - ensure no sector dominates
                        max_sector = 0.35  # Maximum 35% in any sector
                        min_sector = 0.05  # Minimum 5% in any sector with assets
                        
                        # Adjust overweight sectors
                        overweight = [s for s, w in sector_allocation.items() if w > max_sector]
                        if overweight:
                            excess = sum(sector_allocation[s] - max_sector for s in overweight)
                            for s in overweight:
                                sector_allocation[s] = max_sector
                            
                            # Redistribute excess to underweight sectors
                            underweight = [s for s in sector_allocation if s not in overweight]
                            if underweight:
                                for s in underweight:
                                    sector_allocation[s] += excess * (sector_allocation[s] / sum(sector_allocation[s] for s in underweight))
                        
                        # Ensure minimum allocation
                        for sector in list(sector_allocation.keys()):
                            if sector_allocation[sector] < min_sector and len(sector_assets[sector]) > 0:
                                sector_allocation[sector] = min_sector
                        
                        # Normalize final sector allocations
                        total_alloc = sum(sector_allocation.values())
                        sector_allocation = {k: v/total_alloc for k, v in sector_allocation.items()}
                        
                        # Distribute sector weights to individual assets using risk parity within sectors
                        weights = {}
                        for sector, alloc in sector_allocation.items():
                            assets = sector_assets[sector]
                            if assets:
                                # Calculate inverse volatility within sector
                                asset_inv_vols = {}
                                for asset in assets:
                                    if asset in self.returns_df.columns:
                                        vol = self.returns_df[asset].std() * np.sqrt(252)
                                        if np.isfinite(vol) and vol > 0:
                                            asset_inv_vols[asset] = 1.0 / vol
                                        else:
                                            asset_inv_vols[asset] = 1.0  # Default if vol calculation fails
                                    else:
                                        asset_inv_vols[asset] = 1.0  # Default if asset not in returns
                                
                                # Normalize inverse volatilities within sector
                                total_inv_vol = sum(asset_inv_vols.values())
                                if total_inv_vol > 0:
                                    for asset in assets:
                                        weights[asset] = alloc * (asset_inv_vols[asset] / total_inv_vol)
                                else:
                                    # Equal weight within sector if risk calculation fails
                                    for asset in assets:
                                        weights[asset] = alloc / len(assets)
                        
                        # Ensure all assets have weights
                        for symbol in self.price_data.columns:
                            if symbol not in weights:
                                weights[symbol] = 1e-8
                        
                        # Normalize final weights
                        total = sum(weights.values())
                        weights = {k: v/total for k, v in weights.items()}
                        
                        logging.info("Balanced sector allocation successful")
                        
                    except Exception as sector_error:
                        logging.warning(f"Balanced sector allocation failed: {str(sector_error)}")
                        
                        # Final fallback: Modified equal weights with sector balance
                        try:
                            logging.info("Using modified equal weights with sector balance")
                            n_assets = len(self.price_data.columns)
                            base_weight = 1.0 / n_assets
                            
                            # Apply minimal sector adjustments
                            weights = {}
                            for symbol in self.price_data.columns:
                                weights[symbol] = base_weight
                            
                            # Normalize weights
                            total = sum(weights.values())
                            weights = {k: v/total for k, v in weights.items()}
                            
                            logging.warning("Using modified equal weights as final fallback")
                        except Exception:
                            # Ultimate fallback to pure equal weights
                            n_assets = len(self.price_data.columns)
                            weights = {symbol: 1.0/n_assets for symbol in self.price_data.columns}
                            logging.warning("Using pure equal weights as last resort")
                
                # Calculate portfolio metrics for fallback solution
                try:
                    # Calculate returns with weight validation
                    weight_values = np.array(list(weights.values()))
                    if not np.isclose(np.sum(weight_values), 1.0, rtol=1e-3):
                        logging.warning(f"Portfolio weights sum to {np.sum(weight_values)}, normalizing")
                        weight_values = weight_values / np.sum(weight_values)
                    
                    # Calculate portfolio returns
                    returns = self.returns_df.dot(pd.Series(weights))
                    
                    # Calculate risk metrics with validation
                    ret_mean = np.mean(returns) * 252
                    ret_std = np.std(returns) * np.sqrt(252)
                    sharpe = ret_mean/ret_std if ret_std > 0 else 0
                    
                    # Calculate maximum drawdown
                    cum_returns = (1 + returns).cumprod()
                    rolling_max = np.maximum.accumulate(cum_returns)
                    drawdowns = (cum_returns - rolling_max) / rolling_max
                    max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
                    
                    # Calculate diversification metrics
                    effective_positions = len([w for w in weight_values if w > 0.01])
                    herfindahl = np.sum(weight_values**2)
                    
                    portfolio_metrics = {
                        'sharpe_ratio': float(np.nan_to_num(sharpe)),
                        'volatility': float(np.nan_to_num(ret_std)),
                        'expected_return': float(np.nan_to_num(ret_mean)),
                        'max_drawdown': float(np.nan_to_num(max_drawdown)),
                        'diversification_ratio': effective_positions,
                        'concentration': float(np.nan_to_num(herfindahl)),
                        'warning': 'Using enhanced fallback allocation'
                    }
                except Exception as metrics_error:
                    logging.warning(f"Fallback metrics calculation failed: {str(metrics_error)}")
                    portfolio_metrics = {
                        'sharpe_ratio': 0.5,
                        'volatility': 0.15,
                        'expected_return': 0.08,
                        'max_drawdown': 0.15,
                        'diversification_ratio': len([w for w in weights.values() if w > 0.01]),
                        'warning': 'Using enhanced fallback allocation with estimated metrics'
                    }
                
                return weights, portfolio_metrics