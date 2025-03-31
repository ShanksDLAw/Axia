constraints['max_sector'] = min(1.0, constraints.get('max_sector', 0.3) * 2)

# Covariance estimation with robust validation
try:
    # Use Ledoit-Wolf shrinkage for base estimation
    reg_cov_matrix = risk_models.CovarianceShrinkage(
        filtered_price_data,
        frequency=252
    ).ledoit_wolf()
    
    # Basic validation and stabilization
    if not np.all(np.isfinite(reg_cov_matrix)):
        raise ValueError("Invalid covariance matrix")
        
    # Add minimal stability term
    stability_factor = 1e-6
    reg_cov_matrix += np.eye(reg_cov_matrix.shape[0]) * stability_factor
    
except Exception as e:
    logging.warning(f"Covariance estimation failed: {str(e)}")
    reg_cov_matrix = risk_models.sample_cov(filtered_price_data, frequency=252)
    reg_cov_matrix += np.eye(reg_cov_matrix.shape[0]) * 1e-6

# Set up efficient frontier
ef = EfficientFrontier(
    filtered_returns,
    reg_cov_matrix,
    weight_bounds=(0.01, constraints.get('max_position', 0.1))
)

# Add objective function
ef.add_objective(objective_functions.L2_reg, gamma=0.5)

# Add sector constraints if defined
if self.sectors:
    sector_mapper = {asset: self.sectors.get(asset, 'Other') for asset in valid_symbols}
    sector_bounds = {sector: (0.0, constraints.get('max_sector', 0.25)) 
                    for sector in set(sector_mapper.values())}
    ef.add_sector_constraints(sector_mapper, sector_lower={s: v[0] for s, v in sector_bounds.items()},
                            sector_upper={s: v[1] for s, v in sector_bounds.items()})

# Optimize portfolio
try:
    ef.min_volatility()
    weights = ef.clean_weights(cutoff=0.01)
    perf = ef.portfolio_performance()
    metrics = {
        'expected_return': float(perf[0]),
        'volatility': float(perf[1]),
        'sharpe_ratio': float(perf[2]),
        'num_assets': len([w for w in weights.values() if w > 0.01])
    }
except Exception as e:
    logging.error(f"Portfolio optimization failed: {str(e)}")
    weights = {symbol: 1.0/len(valid_symbols) for symbol in valid_symbols}
    metrics = {
        'expected_return': 0.02,
        'volatility': 0.1,
        'sharpe_ratio': 0.2,
        'num_assets': len(valid_symbols),
        'warning': 'Using equal weight fallback'
    }

return weights, metrics