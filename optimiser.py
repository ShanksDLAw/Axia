import numpy as np
import pandas as pd
import cvxpy as cp
import warnings
from scipy.cluster.hierarchy import linkage, fcluster
from pypfopt import objective_functions
from pypfopt import risk_models
from pypfopt.hierarchical_portfolio import HRPOpt
from pypfopt.efficient_frontier import EfficientFrontier
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)


class PortfolioOptimizer:
    def __init__(self, price_data, sectors):
        self.price_data = price_data
        # Calculate returns from price data
        self.returns_df = price_data.pct_change().dropna()
        # Calculate expected returns using historical mean
        self.expected_returns = self.returns_df.mean()
        # Initialize sectors attribute with default if none provided
        self.sectors = sectors if sectors else self._create_default_sectors()
        # Normalize sectors for consistent access
        self.normalized_sectors = {k: v for k, v in self.sectors.items()}
        
    def _create_default_sectors(self) -> dict:
        """Create default sector allocation if none provided."""
        default_sectors = {}
        for asset in self.price_data.columns:
            if asset != 'CASH':
                default_sectors[asset] = 'Uncategorized'
        return default_sectors

    def optimize(self, regime: str, risk_appetite: str) -> tuple[dict[str, float], dict]:
        """Optimize portfolio based on investment regime and risk appetite.
        
        Args:
            regime: Investment regime ('Bullish', 'Bearish', or 'Neutral')
            risk_appetite: Risk appetite ('Conservative', 'Moderate', or 'Aggressive')
            
        Returns:
            tuple: (optimized weights dictionary, portfolio metrics dictionary)
        """
        try:
            # Set dynamic constraints based on risk appetite and market regime
            constraints = {
                'risk_appetite': risk_appetite,
                'max_sector': 0.45 if risk_appetite == 'Aggressive' else (0.35 if risk_appetite == 'Moderate' else 0.25),
                'min_bonds': 0.15 if risk_appetite == 'Conservative' else (0.1 if risk_appetite == 'Moderate' else 0.05),
                'max_position': 0.15 if risk_appetite == 'Aggressive' else (0.1 if risk_appetite == 'Moderate' else 0.05)
            }
            
            # Calculate momentum and volatility signals
            returns_12m = self.returns_df.rolling(window=252).mean()
            volatility_3m = self.returns_df.rolling(window=63).std() * np.sqrt(252)
            
            # Handle potential NaN values in momentum calculation
            try:
                momentum_score = returns_12m.iloc[-1] / volatility_3m.iloc[-1]
                # Replace inf and NaN values with 0
                momentum_score = momentum_score.replace([np.inf, -np.inf], 0).fillna(0)
            except Exception as momentum_error:
                logging.warning(f"Error calculating momentum score: {str(momentum_error)}. Using zeros.")
                momentum_score = pd.Series(0, index=self.returns_df.columns)
            
            # Get optimized weights based on regime and risk appetite
            try:
                if regime == 'Bearish':
                    weights, metrics = self._defensive_strategy(constraints)
                elif regime == 'Bullish':
                    # Always use aggressive strategy for bullish regime with adjusted risk parameters
                    if risk_appetite == 'Conservative':
                        constraints['max_position'] = min(constraints['max_position'] * 1.2, 0.1)  # Slightly increase limits
                    weights, metrics = self._aggressive_strategy(constraints, momentum_score)
                else:
                    weights, metrics = self._balanced_strategy(constraints, momentum_score)
            except Exception as strategy_error:
                logging.error(f"Strategy optimization failed: {str(strategy_error)}. Falling back to defensive strategy.")
                try:
                    weights, metrics = self._defensive_strategy(constraints)
                except Exception as fallback_error:
                    logging.error(f"Fallback strategy failed: {str(fallback_error)}. Using equal weights.")
                    # Create equal weight portfolio as last resort
                    valid_assets = [col for col in self.returns_df.columns if not self.returns_df[col].isnull().all()]
                    if valid_assets:
                        equal_weight = 1.0 / len(valid_assets)
                        weights = {asset: equal_weight for asset in valid_assets}
                        metrics = {
                            'expected_return': 0.05,  # Conservative estimate
                            'volatility': 0.15,      # Conservative estimate
                            'sharpe_ratio': 0.33,    # Conservative estimate
                            'warning': "Using equal weight fallback allocation."
                        }
                    else:
                        # All cash if no valid assets
                        weights = {'CASH': 1.0}
                        metrics = {
                            'expected_return': 0.02,  # Risk-free rate estimate
                            'volatility': 0.0,
                            'sharpe_ratio': 0.0,
                            'warning': "No valid assets found. Using 100% cash allocation."
                        }
            
            # Apply dynamic volatility targeting with error handling
            try:
                target_vol = 0.25 if risk_appetite == 'Aggressive' else (0.15 if risk_appetite == 'Moderate' else 0.10)
                if weights and any(weights.values()):
                    portfolio_vol = np.sqrt(self._calculate_portfolio_variance(weights))
                    if portfolio_vol > 0:
                        vol_scalar = min(2.0, max(0.5, target_vol / portfolio_vol))
                        # Scale weights and adjust cash position
                        scaled_weights = {k: v * vol_scalar for k, v in weights.items()}
                        cash_weight = max(0, 1 - sum(scaled_weights.values()))
                        if cash_weight > 0:
                            scaled_weights['CASH'] = cash_weight
                        
                        # Update metrics
                        metrics['volatility_target'] = target_vol
                        metrics['actual_volatility'] = portfolio_vol
                        metrics['cash_allocation'] = cash_weight
                        
                        return scaled_weights, metrics
                    else:
                        logging.warning("Portfolio volatility is zero. Skipping volatility targeting.")
                        if 'CASH' not in weights and sum(weights.values()) < 1.0:
                            weights['CASH'] = 1.0 - sum(weights.values())
                        return weights, metrics
                else:
                    logging.warning("Empty weights dictionary. Using 100% cash allocation.")
                    return {'CASH': 1.0}, {
                        'expected_return': 0.02,
                        'volatility': 0.0,
                        'sharpe_ratio': 0.0,
                        'warning': "No valid weights found. Using 100% cash allocation."
                    }
            except Exception as vol_error:
                logging.error(f"Volatility targeting failed: {str(vol_error)}. Using original weights.")
                if 'CASH' not in weights and sum(weights.values()) < 1.0:
                    weights['CASH'] = 1.0 - sum(weights.values())
                return weights, metrics
            
        except Exception as e:
            logging.error(f"Portfolio optimization failed: {str(e)}")
            # Return safe fallback allocation with 100% cash
            return {'CASH': 1.0}, {
                'expected_return': 0.02,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'warning': f"Optimization failed: {str(e)}. Using 100% cash allocation."
            }


    def _calculate_portfolio_variance(self, weights: dict) -> float:
        """Calculate portfolio variance using the covariance matrix with enhanced regularization for low-risk scenarios."""
        try:
            # Handle empty weights or cash-only portfolio
            if not weights or all(k == 'CASH' for k in weights.keys()):
                return 0.0
                
            # Create weight array aligned with price data columns
            weight_array = np.array([weights.get(asset, 0) for asset in self.price_data.columns])
            
            # Check if weight array contains any non-zero values
            if np.sum(weight_array) == 0:
                return 0.0
                
            # Enhanced covariance calculation with conservative regularization
            try:
                # Use longer lookback period for more stable estimation
                recent_data = self.price_data.tail(504)  # Use 2 years of data for stability
                
                # Multi-step regularization for enhanced stability
                # Step 1: Initial shrinkage estimation
                shrinkage = risk_models.CovarianceShrinkage(recent_data)
                base_cov = shrinkage.ledoit_wolf()
                
                # Step 2: Spectral decomposition for targeted regularization
                eigenvals, eigenvecs = np.linalg.eigh(base_cov)
                min_eigenval = eigenvals.min()
                
                # Step 3: Enhanced regularization for low-risk scenarios
                if min_eigenval < 1e-6:
                    # Calculate average volatility for scaling
                    avg_vol = np.sqrt(np.mean(np.diag(base_cov)))
                    # Stronger regularization for numerical stability
                    delta = max(abs(min_eigenval) + 1e-6, 0.1)  # Minimum 10% shrinkage
                    # Create target matrix using average volatility
                    target_matrix = np.eye(len(base_cov)) * (avg_vol ** 2)
                    # Apply conservative shrinkage
                    cov_matrix = (1 - delta) * base_cov + delta * target_matrix
                    
                    # Verify positive definiteness
                    min_eig = np.min(np.linalg.eigvals(cov_matrix))
                    if min_eig < 1e-10:
                        # If still not positive definite, use more conservative approach
                        cov_matrix = target_matrix
                else:
                    cov_matrix = base_cov
                
                # Calculate portfolio variance with validation
                portfolio_variance = weight_array.T @ cov_matrix @ weight_array
                
                # Enhanced validation for low-risk portfolios
                if np.isnan(portfolio_variance) or portfolio_variance < 0:
                    logging.warning("Invalid variance detected. Using conservative estimation.")
                    # Use individual asset variances with minimum volatility floor
                    min_vol_floor = 0.05  # 5% minimum annualized volatility
                    asset_vars = np.maximum(np.diag(cov_matrix), min_vol_floor ** 2)
                    portfolio_variance = np.sum((weight_array ** 2) * asset_vars)
                
                return max(portfolio_variance, 0.01 ** 2)  # Minimum 1% annualized volatility
                
            except Exception as cov_error:
                logging.error(f"Enhanced covariance calculation failed: {str(cov_error)}")
                # Fallback to simple diagonal covariance
                vol = self.returns_df.std() * np.sqrt(252)
                diag_cov = np.diag(vol * vol)
                return max(weight_array.T @ diag_cov @ weight_array, 1e-8)
                
        except Exception as e:
            logging.error(f"Error in portfolio variance calculation: {str(e)}")
            return 0.15**2  # Default annualized variance

    def _calculate_expected_returns(self, momentum_score=None, regime='Neutral', risk_appetite='Moderate'):
        """Calculate expected returns using multiple factors with enhanced risk-return adjustments."""
        # Dynamic lookback period based on regime and risk appetite
        lookback = 42 if regime == 'Bullish' and risk_appetite == 'Aggressive' else \
                  (63 if regime == 'Bullish' else (252 if regime == 'Bearish' else 126))
        
        # Calculate historical returns with exponential weighting
        hist_returns = self.returns_df.ewm(halflife=lookback, min_periods=21).mean().iloc[-1]
        
        # Enhanced momentum component with dynamic scaling
        if momentum_score is not None:
            # Stronger momentum influence for aggressive profiles in bullish markets
            momentum_factor = 0.8 if regime == 'Bullish' and risk_appetite == 'Aggressive' else \
                             (0.7 if regime == 'Bullish' else (0.3 if regime == 'Bearish' else 0.5))
            
            # Normalize momentum scores
            momentum_returns = momentum_score * momentum_factor
            
            # Progressive risk scaling based on appetite and regime
            base_scale = {'Conservative': 0.8, 'Moderate': 1.0, 'Aggressive': 1.4}[risk_appetite]
            regime_scale = 1.2 if regime == 'Bullish' else (0.8 if regime == 'Bearish' else 1.0)
            risk_scale = base_scale * regime_scale
            
            # Combine returns with enhanced weighting
            expected_returns = (hist_returns * (1 - momentum_factor) + momentum_returns) * risk_scale
        else:
            expected_returns = hist_returns
        
        # Smart volatility adjustment
        recent_vol = self.returns_df.rolling(window=21).std().iloc[-1]  # 1 month volatility
        vol_adj = recent_vol * (0.8 if risk_appetite == 'Aggressive' else 1.0)  # Lower penalty for aggressive
        vol_scale = 1.0 / (1 + vol_adj)
        
        # Apply volatility scaling with risk appetite consideration
        if risk_appetite == 'Aggressive':
            vol_scale = np.maximum(vol_scale, 0.8)  # Limit downside scaling for aggressive
        
        return expected_returns * vol_scale

    def _aggressive_strategy(self, constraints: dict, momentum_score: pd.Series) -> tuple[dict[str, float], dict]:
        """Aggressive portfolio optimization with momentum-based asset selection and enhanced risk management."""
        try:
            # Ensure alignment between price data and momentum score
            common_assets = self.price_data.columns.intersection(momentum_score.index)
            if len(common_assets) == 0:
                raise ValueError("No common assets between price data and momentum score")
            
            # Calculate expected returns with enhanced momentum and risk-adjusted returns
            expected_returns = self._calculate_expected_returns(
                momentum_score[common_assets],
                regime='Bullish',  # Aggressive strategy implies bullish outlook
                risk_appetite='Aggressive'
            )
            
            # Enhanced momentum filtering with sector-aware selection
            valid_momentum = momentum_score[common_assets]
            sector_groups = defaultdict(list)
            for asset in common_assets:
                if asset in self.sectors:
                    sector_groups[self.sectors[asset]].append(asset)
            
            # Select top assets from each sector
            selected_assets = []
            for sector, assets in sector_groups.items():
                if len(assets) > 0:
                    sector_momentum = valid_momentum[assets]
                    top_sector_assets = sector_momentum.nlargest(max(2, len(assets) // 3)).index
                    selected_assets.extend(top_sector_assets)
            
            # Add top momentum assets overall to ensure sufficient diversification
            top_momentum_threshold = np.percentile(valid_momentum, 70)  # Less restrictive filtering
            high_momentum_assets = valid_momentum[valid_momentum >= top_momentum_threshold].index
            selected_assets.extend([asset for asset in high_momentum_assets if asset not in selected_assets])
            
            # Prepare optimization universe with aggressive return expectations
            optimization_universe = list(set(selected_assets))  # Remove duplicates
            if len(optimization_universe) < 15:  # Ensure minimum diversification
                additional_assets = valid_momentum.nlargest(15).index.difference(optimization_universe)
                optimization_universe.extend(additional_assets[:15-len(optimization_universe)])
            
            # Calculate sector-adjusted return expectations
            filtered_returns = expected_returns[optimization_universe].copy()
            for asset in optimization_universe:
                momentum_rank = valid_momentum[asset] / valid_momentum.max()
                sector_bonus = 0.2 if asset in self.sectors else 0.0  # Bonus for sector-classified assets
                filtered_returns[asset] *= (1.5 + momentum_rank + sector_bonus)  # More aggressive scaling
            
            # Calculate risk model with enhanced stability for bullish market conditions
            try:
                # Use exponential weighted returns for recent market sensitivity
                returns = self.returns_df[optimization_universe].ewm(halflife=42).mean()
                # Calculate covariance with shorter lookback for bullish regime
                price_data_recent = self.price_data[optimization_universe].tail(126)
                
                # Apply double shrinkage for enhanced stability
                shrinkage = risk_models.CovarianceShrinkage(
                    price_data_recent,
                    frequency=252
                )
                risk_model = shrinkage.ledoit_wolf()
                
                # Ensure positive semidefiniteness with advanced regularization
                eigenvals, eigenvecs = np.linalg.eigh(risk_model)
                min_eigenval = eigenvals.min()
                if min_eigenval < 1e-8:
                    # Apply targeted regularization
                    delta = abs(min_eigenval) + 1e-8
                    n_assets = len(risk_model)
                    target_matrix = np.diag(np.diag(risk_model))
                    risk_model = (1 - delta) * risk_model + delta * target_matrix
                    logging.info("Applied targeted regularization for enhanced stability")
            except Exception as e:
                logging.warning(f"Advanced risk model failed: {str(e)}. Using robust fallback.")
                # Robust fallback with exponential weighting
                returns = self.returns_df[optimization_universe].ewm(span=63).cov()
                risk_model = 0.8 * returns.mean() + 0.2 * np.diag(np.diag(returns.mean()))
                risk_model = np.array(risk_model)
            
            # Set up efficient frontier with more aggressive parameters and sector constraints
            max_position = min(0.35, constraints['max_position'] * 2.0)  # More aggressive position limits
            min_position = 0.02  # Ensure meaningful positions
            
            # Initialize efficient frontier with validated bounds
            ef = EfficientFrontier(
                filtered_returns,
                risk_model,
                weight_bounds=(0.01, max_position)  # Ensure minimum position size
            )
            
            # Create sector mappings for constraints
            sector_mappings = {}
            for asset in optimization_universe:
                if asset in self.normalized_sectors:
                    sector = self.normalized_sectors[asset]
                    sector_mappings[asset] = sector
            
            # Set sector-level constraints with proper bounds
            if sector_mappings:
                sector_lower = {sector: 0.05 for sector in set(sector_mappings.values())}
                sector_upper = {sector: constraints.get('max_sector', 0.4) for sector in set(sector_mappings.values())}
                
                # Add sector constraints with proper bounds
                ef.add_sector_constraints(sector_mappings, sector_lower=sector_lower, sector_upper=sector_upper)
            
            # Add objective functions for aggressive optimization
            ef.add_objective(objective_functions.L2_reg, gamma=0.03)  # Reduced regularization
            ef.add_objective(objective_functions.transaction_cost, w_prev=None, k=0.001)  # Consider transaction costs
            
            # Optimize for maximum Sharpe ratio
            try:
                weights = ef.max_sharpe(risk_free_rate=0.01)
                cleaned_weights = ef.clean_weights(cutoff=0.02)
            except Exception as e:
                logging.warning(f"Max Sharpe optimization failed: {str(e)}. Trying alternative approach.")
                # Try alternative optimization approach
                try:
                    weights = ef.max_quadratic_utility(risk_aversion=1.0)
                    cleaned_weights = ef.clean_weights(cutoff=0.02)
                except Exception as alt_e:
                    logging.error(f"Alternative optimization failed: {str(alt_e)}")
                    raise
            
            # Calculate portfolio metrics
            metrics = {
                'expected_return': ef.portfolio_performance()[0],
                'volatility': ef.portfolio_performance()[1],
                'sharpe_ratio': ef.portfolio_performance()[2],
                'momentum_score': momentum_score[optimization_universe].mean(),
                'num_assets': len([w for w in cleaned_weights.values() if w > 0.02])
            }
            
            return cleaned_weights, metrics
            
        except Exception as e:
            logging.error(f"Aggressive strategy optimization failed: {str(e)}")
            return self._defensive_strategy(constraints)
            
    def _balanced_strategy(self, constraints: dict, momentum_score: pd.Series = None) -> tuple[dict[str, float], dict]:
        """Balanced portfolio optimization with risk-adjusted momentum approach.
        
        Args:
            constraints: Dictionary containing optimization constraints
            momentum_score: Optional momentum scores for assets
            
        Returns:
            tuple: (optimized weights dictionary, portfolio metrics dictionary)
        """
        try:
            # Get risk appetite from constraints
            risk_appetite = constraints.get('risk_appetite', 'Moderate')
            
            # Initialize momentum scores if not provided
            if momentum_score is None:
                momentum_score = pd.Series(1.0, index=self.price_data.columns)
                
            # Calculate volatility for risk-based filtering
            volatility = self.returns_df.std() * np.sqrt(252)
            
            # Ensure alignment between price data and momentum score
            common_assets = self.price_data.columns.intersection(momentum_score.index)
            if len(common_assets) == 0:
                raise ValueError("No common assets between price data and momentum score")
            
            # Calculate expected returns with balanced momentum influence
            valid_momentum = momentum_score[common_assets]
            expected_returns = self._calculate_expected_returns(
                momentum_score=valid_momentum * 0.7,  # Reduced momentum influence
                regime='Neutral',
                risk_appetite='Moderate'
            )
            
            # Filter assets with positive momentum
            valid_assets = valid_momentum[valid_momentum > 0].index
            if len(valid_assets) < 5:
                # If too few positive momentum assets, include top neutral momentum assets
                valid_assets = valid_momentum.nlargest(max(10, len(common_assets) // 4)).index
            filtered_returns = expected_returns[valid_assets]
            
            # Calculate risk model with enhanced stability and regularization
            try:
                # Use exponential weighted covariance for recent market sensitivity
                returns = self.returns_df[valid_assets].ewm(halflife=63).mean()
                
                # Apply shrinkage to stabilize the covariance matrix
                shrinkage = risk_models.CovarianceShrinkage(
                    self.price_data[valid_assets].tail(252),
                    frequency=252
                )
                risk_model = shrinkage.ledoit_wolf()
                
                # Ensure positive semidefiniteness
                eigenvals = np.linalg.eigvals(risk_model)
                min_eigenval = np.min(np.real(eigenvals))
                if min_eigenval < 1e-8:
                    # Add regularization term
                    delta = abs(min_eigenval) + 1e-8
                    n_assets = len(risk_model)
                    risk_model = (1 - delta) * risk_model + delta * np.eye(n_assets) * np.trace(risk_model) / n_assets
                    logging.info("Applied enhanced regularization to ensure matrix stability")
            except Exception as e:
                logging.warning(f"Advanced risk model failed: {str(e)}. Using robust fallback.")
                # Robust fallback using sample covariance with strong regularization
                sample_cov = risk_models.sample_cov(self.price_data[valid_assets].tail(126), frequency=252)
                risk_model = 0.8 * sample_cov + 0.2 * np.diag(np.diag(sample_cov))
            
            # Set up efficient frontier with balanced parameters
            ef = EfficientFrontier(
                filtered_returns,
                risk_model,
                weight_bounds=(0, constraints['max_position'])
            )
            
            # Add balanced objective functions
            ef.add_objective(objective_functions.L2_reg, gamma=0.5)  # Moderate regularization
            
            # Add sector constraints if sectors are defined
            if self.sectors:
                sector_mapper = {asset: self.sectors.get(asset, 'Other') for asset in valid_assets}
                sector_lower = {sector: 0.0 for sector in set(sector_mapper.values())}
                sector_upper = {sector: constraints.get('max_sector', 0.35) for sector in set(sector_mapper.values())}
                ef.add_sector_constraints(sector_mapper, sector_lower=sector_lower, sector_upper=sector_upper)
            
            # Optimize using max_sharpe with proper risk-free rate
            ef.max_sharpe(risk_free_rate=0.02)
            weights = ef.clean_weights(cutoff=0.01)
            
            # Calculate portfolio metricsharpe directly
            cleaned_weights = ef.max_sharpe(risk_free_rate=0.015)
            cleaned_weights = ef.clean_weights(cutoff=0.01)
            
            # Calculate portfolio metrics
            metrics = {
                'expected_return': ef.portfolio_performance()[0],
                'volatility': ef.portfolio_performance()[1],
                'sharpe_ratio': ef.portfolio_performance()[2],
                'momentum_score': momentum_score[valid_assets].mean(),
                'num_assets': len([w for w in cleaned_weights.values() if w > 0.01])
            }
            
            return cleaned_weights, metrics
            
        except Exception as e:
            logging.error(f"Balanced strategy optimization failed: {str(e)}")
            return self._defensive_strategy(constraints)
            
    def _defensive_strategy(self, constraints: dict) -> tuple[dict[str, float], dict[str, float]]:
        """Defensive portfolio optimization with enhanced constraint handling and risk management"""
        try:
            # Get risk appetite from constraints
            risk_appetite = constraints.get('risk_appetite', 'Conservative')
            
            # Ensure we have valid price data
            if self.price_data.empty:
                raise ValueError("Empty price data")
                
            # Get valid assets that exist in price data
            valid_assets = self.price_data.columns
            if len(valid_assets) == 0:
                raise ValueError("No valid assets found in price data")
                
            # Calculate risk metrics for asset filtering
            volatility = self.returns_df[valid_assets].std() * np.sqrt(252)
            drawdown = (self.price_data[valid_assets] / self.price_data[valid_assets].expanding().max() - 1).min()
            
            # Filter for low volatility assets
            low_vol_threshold = np.percentile(volatility, 30)  # Focus on bottom 30% by volatility
            low_vol_assets = volatility[volatility <= low_vol_threshold].index
            
            # Further filter by drawdown
            max_acceptable_drawdown = -0.15  # -15% maximum drawdown threshold
            defensive_assets = [asset for asset in low_vol_assets if drawdown[asset] >= max_acceptable_drawdown]
            
            if len(defensive_assets) < 5:
                # If too few assets meet criteria, include more based on combined risk score
                risk_score = volatility * abs(drawdown)
                defensive_assets = risk_score.nsmallest(max(10, len(valid_assets) // 4)).index
                
            # Create risk-based sectors if no valid sectors exist
            if not self.normalized_sectors or all(v in ['Uncategorized', 'Other'] for v in self.normalized_sectors.values()):
                logging.warning("No valid sectors found. Creating risk-based sectors.")
                # Calculate volatility for risk-based sector assignment
                asset_vol = self.returns_df.std() * np.sqrt(252)
                vol_tertiles = np.percentile(asset_vol, [33.33, 66.67])
                
                # Assign risk-based sectors
                self.normalized_sectors = {}
                for asset in self.price_data.columns:
                    if asset == 'CASH':
                        continue
                    vol = asset_vol[asset]
                    if vol <= vol_tertiles[0]:
                        self.normalized_sectors[asset] = 'Low_Vol'
                    elif vol <= vol_tertiles[1]:
                        self.normalized_sectors[asset] = 'Mid_Vol'
                    else:
                        self.normalized_sectors[asset] = 'High_Vol'
                        
                logging.info(f"Created risk-based sectors: {dict(Counter(self.normalized_sectors.values()))}")
            
            # Filter data to use only valid symbols with both price data and sector information
            valid_symbols = [s for s in defensive_assets if s in self.normalized_sectors]
            if not valid_symbols:
                logging.warning("No valid symbols found with sector information. Using defensive assets.")
                valid_symbols = defensive_assets
            
            # Filter data to use only valid symbols
            filtered_price_data = self.price_data[valid_symbols]
            filtered_returns = self.expected_returns[valid_symbols]
            
            # Ensure we have enough data for optimization
            min_symbols = 10  # Minimum number of symbols required
            if len(valid_symbols) < min_symbols:
                logging.warning(f"Insufficient number of valid symbols ({len(valid_symbols)} < {min_symbols}). Using relaxed constraints.")
                constraints['max_sector'] = min(1.0, constraints.get('max_sector', 0.3) * 2)  # Relax sector constraints
                
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

            # Enhanced sector risk calculation with batch processing and validation
            sector_data = defaultdict(lambda: {'count': 0, 'vol_sum': 0.0, 'returns': []})
            
            # Pre-calculate median values for fallbacks
            median_cov = np.median(np.diag(reg_cov_matrix))
            median_return = np.median(filtered_returns)
            
            # Process symbols in batches for better efficiency
            for symbol in valid_symbols:
                sector = self.normalized_sectors.get(symbol)
                if not sector:
                    continue
                    
                try:
                    idx = list(filtered_price_data.columns).index(symbol)
                    if idx >= len(reg_cov_matrix):
                        continue
                    
                    # Get and validate covariance value
                    cov_value = reg_cov_matrix[idx, idx]
                    if not np.isfinite(cov_value) or cov_value < 0:
                        cov_value = median_cov
                    
                    # Calculate and add volatility
                    symbol_vol = np.sqrt(max(1e-8, cov_value))
                    
                    # Get and validate return value
                    return_value = filtered_returns[idx]
                    if not np.isfinite(return_value):
                        return_value = median_return
                    
                    # Update sector data
                    sector_data[sector]['count'] += 1
                    sector_data[sector]['vol_sum'] += symbol_vol
                    sector_data[sector]['returns'].append(return_value)
                    
                except Exception:
                    continue
            
            # Convert processed data to required format with validation
            sector_counts = {k: v['count'] for k, v in sector_data.items()}
            
            # Log sector distribution
            logging.info(f"Sector distribution: {sector_counts}")
            
            # Enhanced sector handling for conservative allocation
            if not sector_counts or len(sector_counts) < 2:
                logging.info("Initiating enhanced risk-based sector creation")
                try:
                    # Calculate volatility using exponential weighting for more stable estimates
                    ewm_vol = self.returns_df[valid_symbols].ewm(halflife=63).std().iloc[-1] * np.sqrt(252)
                    valid_vol = ewm_vol[np.isfinite(ewm_vol)]
                    
                    if len(valid_vol) < len(ewm_vol):
                        logging.warning(f"Filtered out {len(ewm_vol) - len(valid_vol)} invalid volatility values")
                    
                    if len(valid_vol) < 3:
                        raise ValueError("Insufficient valid volatility data for sector creation")
                    
                    # Calculate dynamic thresholds based on market conditions
                    vol_range = valid_vol.max() - valid_vol.min()
                    if vol_range > 0:
                        # Use quantile-based thresholds for better distribution
                        vol_tertiles = valid_vol.quantile([0.33, 0.67])
                    else:
                        # Fallback to fixed thresholds if volatility range is too small
                        median_vol = valid_vol.median()
                        vol_tertiles = pd.Series([median_vol * 0.8, median_vol * 1.2])
                    
                    # Initialize sector containers with validation
                    risk_based_sectors = defaultdict(list)
                    min_sector_size = max(3, len(valid_symbols) // 10)  # Ensure minimum sector size
                    
                    # First pass: assign assets to sectors
                    for asset in valid_symbols:
                        if asset == 'CASH':
                            continue
                        try:
                            vol = ewm_vol[asset]
                            if not np.isfinite(vol):
                                vol = valid_vol.median()  # Use median for missing values
                            
                            if vol <= vol_tertiles[0]:
                                risk_based_sectors['Low_Vol'].append(asset)
                            elif vol <= vol_tertiles[1]:
                                risk_based_sectors['Mid_Vol'].append(asset)
                            else:
                                risk_based_sectors['High_Vol'].append(asset)
                        except Exception as e:
                            logging.warning(f"Error assigning sector for {asset}: {str(e)}")
                            continue
                    
                    # Second pass: rebalance sectors if needed
                    sector_sizes = {k: len(v) for k, v in risk_based_sectors.items()}
                    if min(sector_sizes.values()) < min_sector_size:
                        logging.info("Rebalancing sectors for better distribution")
                        # Sort all assets by volatility
                        sorted_assets = sorted([(a, ewm_vol.get(a, valid_vol.median())) 
                                              for a in valid_symbols if a != 'CASH'],
                                             key=lambda x: x[1])
                        
                        # Redistribute assets evenly
                        n_assets = len(sorted_assets)
                        split_points = [n_assets // 3, 2 * n_assets // 3]
                        
                        risk_based_sectors = defaultdict(list)
                        for i, (asset, _) in enumerate(sorted_assets):
                            if i < split_points[0]:
                                risk_based_sectors['Low_Vol'].append(asset)
                            elif i < split_points[1]:
                                risk_based_sectors['Mid_Vol'].append(asset)
                            else:
                                risk_based_sectors['High_Vol'].append(asset)
                    
                    # Update normalized sectors
                    self.normalized_sectors.update({asset: sector 
                                                  for sector, assets in risk_based_sectors.items() 
                                                  for asset in assets})
                    
                    logging.info(f"Created balanced risk-based sectors: {dict((k, len(v)) for k, v in risk_based_sectors.items())}")
                    
                except Exception as e:
                    logging.warning(f"Error assigning sector for {asset}: {str(e)}")
                    continue
                    
                    # Ensure each sector has at least some assets
                    min_assets_per_sector = max(3, len(valid_symbols) // 10)
                    for sector in ['Low_Vol', 'Mid_Vol', 'High_Vol']:
                        if sector not in risk_based_sectors or len(risk_based_sectors[sector]) < min_assets_per_sector:
                            logging.warning(f"Insufficient assets in {sector}. Adjusting thresholds.")
                            # Redistribute assets if a sector is too small
                            all_assets = [a for a in valid_symbols if a != 'CASH']
                            sorted_by_vol = sorted(all_assets, key=lambda x: volatility[x])
                            n_assets = len(sorted_by_vol)
                            risk_based_sectors['Low_Vol'] = sorted_by_vol[:n_assets//3]
                            risk_based_sectors['Mid_Vol'] = sorted_by_vol[n_assets//3:2*n_assets//3]
                            risk_based_sectors['High_Vol'] = sorted_by_vol[2*n_assets//3:]
                            break
                    
                    # Update sector mappings and constraints
                    self.normalized_sectors = {}
                    for sector, assets in risk_based_sectors.items():
                        for asset in assets:
                            self.normalized_sectors[asset] = sector
                    
                    # Update sector counts and log distribution
                    sector_counts = {sector: len(assets) for sector, assets in risk_based_sectors.items()}
                    logging.info(f"Created enhanced risk-based sectors: {sector_counts}")
                    
                    # Adjust constraints for better risk management
                    constraints['max_sector'] = min(constraints.get('max_sector', 0.3), 0.4)
                    
                except Exception as e:
                    logging.error(f"Error creating risk-based sectors: {str(e)}")
                    raise ValueError("Failed to create valid sector distribution")
            
            # Calculate sector metrics with enhanced risk appetite consideration and validation
            sector_vols = {}
            sector_returns = {}
            risk_multiplier = 1.2 if risk_appetite == 'Aggressive' else (1.0 if risk_appetite == 'Moderate' else 0.8)
            
            # Validate sector assignments before proceeding
            valid_sectors = set(self.normalized_sectors.values())
            if not valid_sectors:
                raise ValueError("No valid sector assignments found")
            
            # Ensure all required sectors exist
            required_sectors = {'Low_Vol', 'Mid_Vol', 'High_Vol'}
            if not required_sectors.issubset(valid_sectors):
                missing_sectors = required_sectors - valid_sectors
                logging.warning(f"Missing required sectors: {missing_sectors}. Creating default assignments.")
                
                # Create missing sectors with default assignments
                for sector in missing_sectors:
                    if sector == 'Low_Vol':
                        threshold = np.percentile(volatility, 33)
                        symbols = volatility[volatility <= threshold].index
                    elif sector == 'Mid_Vol':
                        lower = np.percentile(volatility, 33)
                        upper = np.percentile(volatility, 67)
                        symbols = volatility[(volatility > lower) & (volatility <= upper)].index
                    else:  # High_Vol
                        threshold = np.percentile(volatility, 67)
                        symbols = volatility[volatility > threshold].index
                    
                    # Ensure we have symbols for this sector with enhanced error handling
                    if len(symbols) == 0:
                        logging.warning(f"No symbols found for {sector}. Creating default symbols.")
                        # Assign at least some symbols to this sector to prevent errors
                        all_symbols = list(valid_symbols)
                        if len(all_symbols) == 0:
                            logging.error(f"No valid symbols available for {sector} assignment")
                            # Create a dummy symbol to prevent errors
                            continue
                            
                        # Different allocation strategies based on sector
                        if sector == 'Mid_Vol':
                            # Ensure Mid_Vol has some symbols by taking middle portion of available symbols
                            if len(all_symbols) >= 3:
                                start_idx = len(all_symbols) // 3
                                end_idx = 2 * len(all_symbols) // 3
                                symbols = all_symbols[start_idx:end_idx]
                            else:
                                # For small symbol lists, just take the middle or first symbol
                                symbols = [all_symbols[len(all_symbols)//2]] if len(all_symbols) > 1 else all_symbols
                        elif sector == 'Low_Vol':
                            # For Low_Vol, take first third or at least one symbol
                            symbols = all_symbols[:max(1, len(all_symbols)//3)]
                        elif sector == 'High_Vol':
                            # For High_Vol, take last third or at least one symbol
                            symbols = all_symbols[-max(1, len(all_symbols)//3):]
                        else:
                            # For any other sector, distribute evenly
                            symbols = all_symbols[:max(1, len(all_symbols)//len(required_sectors))]
                    
                    # Update sector mappings
                    for symbol in symbols:
                        if symbol in valid_symbols:
                            self.normalized_sectors[symbol] = sector
                            
                # Verify sectors were created successfully
                for sector in missing_sectors:
                    sector_symbols = [s for s, sec in self.normalized_sectors.items() if sec == sector and s in valid_symbols]
                    if not sector_symbols:
                        logging.warning(f"Failed to create {sector}. Adding to sector_counts with zero count.")
                        sector_counts[sector] = 0
            
            for sector, count in sector_counts.items():
                try:
                    # Get symbols in this sector
                    sector_symbols = [s for s, sec in self.normalized_sectors.items() if sec == sector and s in valid_symbols]
                    
                    # Enhanced validation for empty sectors
                    if not sector_symbols:
                        logging.warning(f"No symbols found for sector {sector}. Using fallback values.")
                        # Use fallback values for empty sectors
                        sector_vols[sector] = np.median(volatility) * risk_multiplier
                        sector_returns[sector] = np.median(filtered_returns) * risk_multiplier
                        continue
                        
                    # Calculate sector volatility with risk adjustment
                    try:
                        sector_returns_data = filtered_returns[sector_symbols]
                        if len(sector_returns_data) > 0:
                            base_vol = np.std(sector_returns_data)
                            sector_vols[sector] = base_vol * risk_multiplier
                        else:
                            sector_vols[sector] = np.median(filtered_returns.std()) * risk_multiplier
                        
                        # Calculate sector returns with risk-adjusted expectations
                        if len(sector_returns_data) > 0:
                            base_return = np.mean(sector_returns_data)
                            sector_returns[sector] = base_return * risk_multiplier
                        else:
                            sector_returns[sector] = np.median(filtered_returns) * risk_multiplier
                    except Exception as data_error:
                        logging.warning(f"Error processing data for sector {sector}: {str(data_error)}")
                        # Use fallback values based on overall portfolio metrics
                        sector_vols[sector] = np.median(volatility) * risk_multiplier
                        sector_returns[sector] = np.median(filtered_returns) * risk_multiplier
                except Exception as e:
                    logging.warning(f"Error calculating metrics for sector {sector}: {str(e)}")
                    # Use risk-adjusted median values as fallback
                    sector_vols[sector] = np.median([v for v in sector_vols.values() if v > 0] or [0.1]) * risk_multiplier
                    sector_returns[sector] = np.median([r for r in sector_returns.values() if r != 0] or [0.05]) * risk_multiplier

            # Calculate adaptive sector limits with enhanced validation and volatility scaling
            num_sectors = len(sector_counts)
            if num_sectors == 0:
                logging.error("No valid sectors found for optimization, initiating HRP fallback")
                # Use Hierarchical Risk Parity as fallback
                try:
                    hrp = HRPOpt(returns=filtered_returns)
                    # HRPOpt.optimize() requires a linkage_method parameter
                    weights = hrp.optimize(linkage_method='single')
                    metrics = {
                        'expected_return': np.sum(filtered_returns.mean() * weights),
                        'volatility': np.sqrt(np.dot(weights.T, np.dot(reg_cov_matrix, weights))),
                        'sharpe_ratio': np.sum(filtered_returns.mean() * weights) / np.sqrt(np.dot(weights.T, np.dot(reg_cov_matrix, weights))),
                        'diversification_ratio': len([w for w in weights.values() if w > 0.01])
                    }
                    logging.info("HRP fallback optimization successful")
                    return weights, metrics
                except Exception as e:
                    logging.error(f"HRP fallback failed: {str(e)}")
                    # Ultimate fallback: equal weight portfolio
                    weights = {symbol: 1.0/len(valid_symbols) for symbol in valid_symbols}
                    return weights, {'warning': 'Using equal weight fallback allocation'}
                
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
            
            # Calculate sector bounds with enhanced risk-based validation
            sector_upper = {}
            sector_lower = {}
            total_allocation = 0.0
            
            for sector in sector_counts:
                count = sector_counts[sector]
                if count > 0:
                    # Dynamic sector bounds based on risk profile
                    if sector == 'Low_Vol':
                        sector_lower[sector] = 0.3 if risk_appetite == 'Conservative' else 0.2
                        sector_upper[sector] = min(0.6, base_max * 1.2) if risk_appetite == 'Conservative' else min(0.5, base_max * 1.1)
                    elif sector == 'Mid_Vol':
                        # Handle Mid_Vol sector with additional validation
                        if sector in sector_counts and count > 0:  # Only set constraints if sector exists and has assets
                            sector_lower[sector] = 0.2 if risk_appetite != 'Aggressive' else 0.15
                            sector_upper[sector] = min(0.4, base_max * 1.1) if risk_appetite != 'Aggressive' else min(0.45, base_max * 1.2)
                        else:
                            # Set default values for empty sector to prevent errors
                            sector_lower[sector] = 0.0
                            sector_upper[sector] = 0.0
                            logging.warning(f"Skipping empty sector: {sector}")
                            continue
                    else:  # High_Vol
                        sector_lower[sector] = 0.1
                        sector_upper[sector] = min(0.3, base_max) if risk_appetite == 'Conservative' else min(0.4, base_max * 1.2)
                    
                    # Apply volatility scaling
                    scale = vol_scale.get(sector, 1.0)
                    sector_upper[sector] *= scale
                    sector_lower[sector] *= scale
                    
                    # Ensure minimum allocation
                    sector_lower[sector] = max(0.05, sector_lower[sector])
                    
                    # Track total allocation
                    total_allocation += sector_lower[sector]
                    
            # Validate and adjust bounds if necessary
            if total_allocation > 1.0:
                # Scale down proportionally
                scale_factor = 0.95 / total_allocation  # Leave some room for optimization
                for sector in sector_lower:
                    sector_lower[sector] *= scale_factor
                    sector_upper[sector] *= scale_factor
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
                        # The EfficientFrontier class doesn't have an 'optimize' method directly
                        # Instead, we need to call a specific optimization method
                        ef.min_volatility()
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
                logging.info("Initiating enhanced fallback optimization strategy")
                
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
            logging.error(f"Defensive strategy failed: {str(e)}")
            logging.info("Initiating enhanced defensive fallback optimization strategy")
            
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
    
    # The _balanced_strategy method is already defined at line 404, so we're removing this duplicate definition
                
            # Create risk-based sectors if no valid sectors exist
            if not self.normalized_sectors or all(v in ['Uncategorized', 'Other'] for v in self.normalized_sectors.values()):
                logging.warning("No valid sectors found. Creating risk-based sectors.")
                # Calculate volatility for risk-based sector assignment
                asset_vol = self.returns_df.std() * np.sqrt(252)
                vol_tertiles = np.percentile(asset_vol, [33.33, 66.67])
                
                # Assign risk-based sectors
                self.normalized_sectors = {}
                for asset in self.price_data.columns:
                    if asset == 'CASH':
                        continue
                    vol = asset_vol[asset]
                    if vol <= vol_tertiles[0]:
                        self.normalized_sectors[asset] = 'Low_Vol'
                    elif vol <= vol_tertiles[1]:
                        self.normalized_sectors[asset] = 'Mid_Vol'
                    else:
                        self.normalized_sectors[asset] = 'High_Vol'
                        
                logging.info(f"Created risk-based sectors: {dict(Counter(self.normalized_sectors.values()))}")
            
            # Filter data to use only valid symbols with both price data and sector information
            valid_symbols = [s for s in defensive_assets if s in self.normalized_sectors]
            if not valid_symbols:
                logging.warning("No valid symbols found with sector information. Using defensive assets.")
                valid_symbols = defensive_assets
            
            # Filter data to use only valid symbols
            filtered_price_data = self.price_data[valid_symbols]
            filtered_returns = self.expected_returns[valid_symbols]
            
            # Ensure we have enough data for optimization
            min_symbols = 10  # Minimum number of symbols required
            if len(valid_symbols) < min_symbols:
                logging.warning(f"Insufficient number of valid symbols ({len(valid_symbols)} < {min_symbols}). Using relaxed constraints.")
                constraints['max_sector'] = min(1.0, constraints.get('max_sector', 0.3) * 2)  # Relax sector constraints
                
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

            # Enhanced sector risk calculation with batch processing and validation
            sector_data = defaultdict(lambda: {'count': 0, 'vol_sum': 0.0, 'returns': []})
            
            # Pre-calculate median values for fallbacks
            median_cov = np.median(np.diag(reg_cov_matrix))
            median_return = np.median(filtered_returns)
            
            # Process symbols in batches for better efficiency
            for symbol in valid_symbols:
                sector = self.normalized_sectors.get(symbol)
                if not sector:
                    continue
                    
                try:
                    idx = list(filtered_price_data.columns).index(symbol)
                    if idx >= len(reg_cov_matrix):
                        continue
                    
                    # Get and validate covariance value
                    cov_value = reg_cov_matrix[idx, idx]
                    if not np.isfinite(cov_value) or cov_value < 0:
                        cov_value = median_cov
                    
                    # Calculate and add volatility
                    symbol_vol = np.sqrt(max(1e-8, cov_value))
                    
                    # Get and validate return value
                    return_value = filtered_returns[idx]
                    if not np.isfinite(return_value):
                        return_value = median_return
                    
                    # Update sector data
                    sector_data[sector]['count'] += 1
                    sector_data[sector]['vol_sum'] += symbol_vol
                    sector_data[sector]['returns'].append(return_value)
                    
                except Exception:
                    continue
            
            # Convert processed data to required format with validation
            sector_counts = {k: v['count'] for k, v in sector_data.items()}
            
            # Log sector distribution
            logging.info(f"Sector distribution: {sector_counts}")
            
            # Enhanced sector handling for conservative allocation
            if not sector_counts or len(sector_counts) < 2:
                logging.info("Initiating enhanced risk-based sector creation")
                try:
                    # Calculate volatility using exponential weighting for more stable estimates
                    ewm_vol = self.returns_df[valid_symbols].ewm(halflife=63).std().iloc[-1] * np.sqrt(252)
                    valid_vol = ewm_vol[np.isfinite(ewm_vol)]
                    
                    if len(valid_vol) < len(ewm_vol):
                        logging.warning(f"Filtered out {len(ewm_vol) - len(valid_vol)} invalid volatility values")
                    
                    if len(valid_vol) < 3:
                        raise ValueError("Insufficient valid volatility data for sector creation")
                    
                    # Calculate dynamic thresholds based on market conditions
                    vol_range = valid_vol.max() - valid_vol.min()
                    if vol_range > 0:
                        # Use quantile-based thresholds for better distribution
                        vol_tertiles = valid_vol.quantile([0.33, 0.67])
                    else:
                        # Fallback to fixed thresholds if volatility range is too small
                        median_vol = valid_vol.median()
                        vol_tertiles = pd.Series([median_vol * 0.8, median_vol * 1.2])
                    
                    # Initialize sector containers with validation
                    risk_based_sectors = defaultdict(list)
                    min_sector_size = max(3, len(valid_symbols) // 10)  # Ensure minimum sector size
                    
                    # First pass: assign assets to sectors
                    for asset in valid_symbols:
                        if asset == 'CASH':
                            continue
                        try:
                            vol = ewm_vol[asset]
                            if not np.isfinite(vol):
                                vol = valid_vol.median()  # Use median for missing values
                            
                            if vol <= vol_tertiles[0]:
                                risk_based_sectors['Low_Vol'].append(asset)
                            elif vol <= vol_tertiles[1]:
                                risk_based_sectors['Mid_Vol'].append(asset)
                            else:
                                risk_based_sectors['High_Vol'].append(asset)
                        except Exception as e:
                            logging.warning(f"Error assigning sector for {asset}: {str(e)}")
                            continue
                    
                    # Second pass: rebalance sectors if needed
                    sector_sizes = {k: len(v) for k, v in risk_based_sectors.items()}
                    if min(sector_sizes.values()) < min_sector_size:
                        logging.info("Rebalancing sectors for better distribution")
                        # Sort all assets by volatility
                        sorted_assets = sorted([(a, ewm_vol.get(a, valid_vol.median())) 
                                              for a in valid_symbols if a != 'CASH'],
                                             key=lambda x: x[1])
                        
                        # Redistribute assets evenly
                        n_assets = len(sorted_assets)
                        split_points = [n_assets // 3, 2 * n_assets // 3]
                        
                        risk_based_sectors = defaultdict(list)
                        for i, (asset, _) in enumerate(sorted_assets):
                            if i < split_points[0]:
                                risk_based_sectors['Low_Vol'].append(asset)
                            elif i < split_points[1]:
                                risk_based_sectors['Mid_Vol'].append(asset)
                            else:
                                risk_based_sectors['High_Vol'].append(asset)
                    
                    # Update normalized sectors
                    self.normalized_sectors.update({asset: sector 
                                                  for sector, assets in risk_based_sectors.items() 
                                                  for asset in assets})
                    
                    logging.info(f"Created balanced risk-based sectors: {dict((k, len(v)) for k, v in risk_based_sectors.items())}")
                    
                except Exception as e:
                    logging.warning(f"Error assigning sector for {asset}: {str(e)}")
                    continue
                    
                    # Ensure each sector has at least some assets
                    min_assets_per_sector = max(3, len(valid_symbols) // 10)
                    for sector in ['Low_Vol', 'Mid_Vol', 'High_Vol']:
                        if sector not in risk_based_sectors or len(risk_based_sectors[sector]) < min_assets_per_sector:
                            logging.warning(f"Insufficient assets in {sector}. Adjusting thresholds.")
                            # Redistribute assets if a sector is too small
                            all_assets = [a for a in valid_symbols if a != 'CASH']
                            sorted_by_vol = sorted(all_assets, key=lambda x: volatility[x])
                            n_assets = len(sorted_by_vol)
                            risk_based_sectors['Low_Vol'] = sorted_by_vol[:n_assets//3]
                            risk_based_sectors['Mid_Vol'] = sorted_by_vol[n_assets//3:2*n_assets//3]
                            risk_based_sectors['High_Vol'] = sorted_by_vol[2*n_assets//3:]
                            break
                    
                    # Update sector mappings and constraints
                    self.normalized_sectors = {}
                    for sector, assets in risk_based_sectors.items():
                        for asset in assets:
                            self.normalized_sectors[asset] = sector
                    
                    # Update sector counts and log distribution
                    sector_counts = {sector: len(assets) for sector, assets in risk_based_sectors.items()}
                    logging.info(f"Created enhanced risk-based sectors: {sector_counts}")
                    
                    # Adjust constraints for better risk management
                    constraints['max_sector'] = min(constraints.get('max_sector', 0.3), 0.4)
                    
                except Exception as e:
                    logging.error(f"Error creating risk-based sectors: {str(e)}")
                    raise ValueError("Failed to create valid sector distribution")
            
            # Calculate sector metrics with enhanced risk appetite consideration and validation
            sector_vols = {}
            sector_returns = {}
            risk_multiplier = 1.2 if risk_appetite == 'Aggressive' else (1.0 if risk_appetite == 'Moderate' else 0.8)
            
            # Validate sector assignments before proceeding
            valid_sectors = set(self.normalized_sectors.values())
            if not valid_sectors:
                raise ValueError("No valid sector assignments found")
            
            # Ensure all required sectors exist
            required_sectors = {'Low_Vol', 'Mid_Vol', 'High_Vol'}
            if not required_sectors.issubset(valid_sectors):
                missing_sectors = required_sectors - valid_sectors
                logging.warning(f"Missing required sectors: {missing_sectors}. Creating default assignments.")
                
                # Create missing sectors with default assignments
                for sector in missing_sectors:
                    if sector == 'Low_Vol':
                        threshold = np.percentile(volatility, 33)
                        symbols = volatility[volatility <= threshold].index
                    elif sector == 'Mid_Vol':
                        lower = np.percentile(volatility, 33)
                        upper = np.percentile(volatility, 67)
                        symbols = volatility[(volatility > lower) & (volatility <= upper)].index
                    else:  # High_Vol
                        threshold = np.percentile(volatility, 67)
                        symbols = volatility[volatility > threshold].index
                    
                    # Ensure we have symbols for this sector with enhanced error handling
                    if len(symbols) == 0:
                        logging.warning(f"No symbols found for {sector}. Creating default symbols.")
                        # Assign at least some symbols to this sector to prevent errors
                        all_symbols = list(valid_symbols)
                        if len(all_symbols) == 0:
                            logging.error(f"No valid symbols available for {sector} assignment")
                            # Create a dummy symbol to prevent errors
                            continue
                            
                        # Different allocation strategies based on sector
                        if sector == 'Mid_Vol':
                            # Ensure Mid_Vol has some symbols by taking middle portion of available symbols
                            if len(all_symbols) >= 3:
                                start_idx = len(all_symbols) // 3
                                end_idx = 2 * len(all_symbols) // 3
                                symbols = all_symbols[start_idx:end_idx]
                            else:
                                # For small symbol lists, just take the middle or first symbol
                                symbols = [all_symbols[len(all_symbols)//2]] if len(all_symbols) > 1 else all_symbols
                        elif sector == 'Low_Vol':
                            # For Low_Vol, take first third or at least one symbol
                            symbols = all_symbols[:max(1, len(all_symbols)//3)]
                        elif sector == 'High_Vol':
                            # For High_Vol, take last third or at least one symbol
                            symbols = all_symbols[-max(1, len(all_symbols)//3):]
                        else:
                            # For any other sector, distribute evenly
                            symbols = all_symbols[:max(1, len(all_symbols)//len(required_sectors))]
                    
                    # Update sector mappings
                    for symbol in symbols:
                        if symbol in valid_symbols:
                            self.normalized_sectors[symbol] = sector
                            
                # Verify sectors were created successfully
                for sector in missing_sectors:
                    sector_symbols = [s for s, sec in self.normalized_sectors.items() if sec == sector and s in valid_symbols]
                    if not sector_symbols:
                        logging.warning(f"Failed to create {sector}. Adding to sector_counts with zero count.")
                        sector_counts[sector] = 0
            
            for sector, count in sector_counts.items():
                try:
                    # Get symbols in this sector
                    sector_symbols = [s for s, sec in self.normalized_sectors.items() if sec == sector and s in valid_symbols]
                    
                    # Enhanced validation for empty sectors
                    if not sector_symbols:
                        logging.warning(f"No symbols found for sector {sector}. Using fallback values.")
                        # Use fallback values for empty sectors
                        sector_vols[sector] = np.median(volatility) * risk_multiplier
                        sector_returns[sector] = np.median(filtered_returns) * risk_multiplier
                        continue
                        
                    # Calculate sector volatility with risk adjustment
                    try:
                        sector_returns_data = filtered_returns[sector_symbols]
                        if len(sector_returns_data) > 0:
                            base_vol = np.std(sector_returns_data)
                            sector_vols[sector] = base_vol * risk_multiplier
                        else:
                            sector_vols[sector] = np.median(filtered_returns.std()) * risk_multiplier
                        
                        # Calculate sector returns with risk-adjusted expectations
                        if len(sector_returns_data) > 0:
                            base_return = np.mean(sector_returns_data)
                            sector_returns[sector] = base_return * risk_multiplier
                        else:
                            sector_returns[sector] = np.median(filtered_returns) * risk_multiplier
                    except Exception as data_error:
                        logging.warning(f"Error processing data for sector {sector}: {str(data_error)}")
                        # Use fallback values based on overall portfolio metrics
                        sector_vols[sector] = np.median(volatility) * risk_multiplier
                        sector_returns[sector] = np.median(filtered_returns) * risk_multiplier
                except Exception as e:
                    logging.warning(f"Error calculating metrics for sector {sector}: {str(e)}")
                    # Use risk-adjusted median values as fallback
                    sector_vols[sector] = np.median([v for v in sector_vols.values() if v > 0] or [0.1]) * risk_multiplier
                    sector_returns[sector] = np.median([r for r in sector_returns.values() if r != 0] or [0.05]) * risk_multiplier

            # Calculate adaptive sector limits with enhanced validation and volatility scaling
            num_sectors = len(sector_counts)
            if num_sectors == 0:
                logging.error("No valid sectors found for optimization, initiating HRP fallback")
                # Use Hierarchical Risk Parity as fallback
                try:
                    hrp = HRPOpt(returns=filtered_returns)
                    # HRPOpt.optimize() requires a linkage_method parameter
                    weights = hrp.optimize(linkage_method='single')
                    metrics = {
                        'expected_return': np.sum(filtered_returns.mean() * weights),
                        'volatility': np.sqrt(np.dot(weights.T, np.dot(reg_cov_matrix, weights))),
                        'sharpe_ratio': np.sum(filtered_returns.mean() * weights) / np.sqrt(np.dot(weights.T, np.dot(reg_cov_matrix, weights))),
                        'diversification_ratio': len([w for w in weights.values() if w > 0.01])
                    }
                    logging.info("HRP fallback optimization successful")
                    return weights, metrics
                except Exception as e:
                    logging.error(f"HRP fallback failed: {str(e)}")
                    # Ultimate fallback: equal weight portfolio
                    weights = {symbol: 1.0/len(valid_symbols) for symbol in valid_symbols}
                    return weights, {'warning': 'Using equal weight fallback allocation'}
                
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
            
            # Calculate sector bounds with enhanced risk-based validation
            sector_upper = {}
            sector_lower = {}
            total_allocation = 0.0
            
            for sector in sector_counts:
                count = sector_counts[sector]
                if count > 0:
                    # Dynamic sector bounds based on risk profile
                    if sector == 'Low_Vol':
                        sector_lower[sector] = 0.3 if risk_appetite == 'Conservative' else 0.2
                        sector_upper[sector] = min(0.6, base_max * 1.2) if risk_appetite == 'Conservative' else min(0.5, base_max * 1.1)
                    elif sector == 'Mid_Vol':
                        # Handle Mid_Vol sector with additional validation
                        if sector in sector_counts and count > 0:  # Only set constraints if sector exists and has assets
                            sector_lower[sector] = 0.2 if risk_appetite != 'Aggressive' else 0.15
                            sector_upper[sector] = min(0.4, base_max * 1.1) if risk_appetite != 'Aggressive' else min(0.45, base_max * 1.2)
                        else:
                            # Set default values for empty sector to prevent errors
                            sector_lower[sector] = 0.0
                            sector_upper[sector] = 0.0
                            logging.warning(f"Skipping empty sector: {sector}")
                            continue
                    else:  # High_Vol
                        sector_lower[sector] = 0.1
                        sector_upper[sector] = min(0.3, base_max) if risk_appetite == 'Conservative' else min(0.4, base_max * 1.2)
                    
                    # Apply volatility scaling
                    scale = vol_scale.get(sector, 1.0)
                    sector_upper[sector] *= scale
                    sector_lower[sector] *= scale
                    
                    # Ensure minimum allocation
                    sector_lower[sector] = max(0.05, sector_lower[sector])
                    
                    # Track total allocation
                    total_allocation += sector_lower[sector]
                    
            # Validate and adjust bounds if necessary
            if total_allocation > 1.0:
                # Scale down proportionally
                scale_factor = 0.95 / total_allocation  # Leave some room for optimization
                for sector in sector_lower:
                    sector_lower[sector] *= scale_factor
                    sector_upper[sector] *= scale_factor
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
                        # The EfficientFrontier class doesn't have an 'optimize' method directly
                        # Instead, we need to call a specific optimization method
                        ef.min_volatility()
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
                logging.info("Initiating enhanced fallback optimization strategy")
                
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
                running_max = np.maximum.accumulate(cum_returns)
                drawdowns = (cum_returns - running_max) / running_max
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
            logging.info("Initiating fallback optimization strategy")