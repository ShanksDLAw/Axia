import streamlit as st
import pandas as pd
import yfinance as yf
import time
from asset_loader import AssetLoader 
from market_analyser import MarketAnalyzer
from optimiser import PortfolioOptimizer
from backtester import Backtester
from visualization import create_portfolio_dashboard


st.set_page_config(page_title="Axia", layout="wide")
st.title("📊 Axia: Smart Portfolio Optimizer")

# Step 1: Get User Profile
st.subheader("Investor Profile")
risk_level = st.radio("What is your risk tolerance?", ['Low', 'Medium', 'High'], key='risk_level')
horizon = st.selectbox("Investment horizon?", ['Short (1–2 yrs)', 'Medium (3–5 yrs)', 'Long (5+ yrs)'], key='horizon')
goal = st.radio("Investment goal?", ['Growth', 'Income', 'Balanced'], key='goal')

def recommend_strategy(risk, horizon, goal):
    if risk == 'High' and goal == 'Growth':
        return 'Bullish'
    elif risk == 'Low' and goal == 'Income':
        return 'Bearish'
    else:
        return 'Neutral'

regime = recommend_strategy(risk_level, horizon, goal)
st.info(f"📌 Based on your profile, we recommend a **{regime} strategy**.")

if st.button("Run Analysis"):
    # Step 2: Load Assets
    st.subheader("Loading Assets")
    loader = AssetLoader()
    
    # Initialize progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Fetch assets with progress tracking
    status_text.text("Initializing asset loader...")
    progress_bar.progress(10)
    
    status_text.text("Fetching asset list...")
    progress_bar.progress(20)
    symbols, sectors = loader.fetch_assets()
    
    if not symbols:
        progress_bar.empty()
        status_text.empty()
        st.error("No symbols loaded.")
        st.stop()
    
    total_assets = len(symbols)
    status_text.text(f"Successfully loaded {total_assets} assets")
    progress_bar.progress(40)

    # Step 3: Market Data
    st.subheader("Market Data")
    status_text.text("Downloading market data...")
    data = yf.download(symbols, period="5y", group_by="ticker", progress=False)
    progress_bar.progress(60)
    
    price_data = data.xs("Close", level=1, axis=1).dropna(axis=1, how="all") if isinstance(data.columns, pd.MultiIndex) else data['Close'].to_frame().dropna(axis=1, how="all")
    status_text.text(f"Downloaded {price_data.shape[1]} assets with {price_data.shape[0]} data points")
    progress_bar.progress(70)

    # Step 4: Optimization
    st.subheader("Portfolio Optimization")
    status_text.text("Optimizing portfolio...")
    optimizer = PortfolioOptimizer(price_data, sectors)
    
    # Get risk appetite from user profile
    risk_appetite = 'Conservative' if risk_level == 'Low' else ('Aggressive' if risk_level == 'High' else 'Moderate')
    weights, portfolio_metrics = optimizer.optimize(regime, risk_appetite)
    progress_bar.progress(85)
    
    # Check if there's a warning in the portfolio metrics
    if 'warning' in portfolio_metrics:
        st.warning(portfolio_metrics['warning'])
    
    # Display portfolio weights with improved formatting
    weights_df = pd.DataFrame.from_dict(weights, orient="index", columns=["Weight"])
    weights_df = weights_df.sort_values(by="Weight", ascending=False)
    weights_df["Weight %"] = weights_df["Weight"].apply(lambda x: f"{x:.2%}")
    st.dataframe(weights_df)
    
    # Display portfolio metrics with improved formatting and additional metrics
    st.subheader("Portfolio Metrics")
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    
    with metrics_col1:
        st.metric("Expected Return", f"{portfolio_metrics.get('expected_return', 0.0):.2%}")
        st.metric("Diversification", f"{int(portfolio_metrics.get('effective_positions', portfolio_metrics.get('diversification_ratio', 0)))} assets")
    
    with metrics_col2:
        st.metric("Volatility", f"{portfolio_metrics.get('volatility', 0.0):.2%}")
        st.metric("Max Drawdown", f"{portfolio_metrics.get('max_drawdown', 0.0):.2%}")
    
    with metrics_col3:
        st.metric("Sharpe Ratio", f"{portfolio_metrics.get('sharpe_ratio', 0.0):.2f}")
        
        # Add risk level indicator
        risk_level_map = {"Low": "🟢", "Medium": "🟠", "High": "🔴"}
        st.metric("Risk Level", f"{risk_level_map.get(risk_level, '⚪')} {risk_level}")

    # Step 5: Backtest
    st.subheader("Backtest Performance")
    status_text.text("Running backtest analysis...")
    backtester = Backtester(price_data, sectors)
    results = backtester.run_backtest(weights)
    progress_bar.progress(100)
    status_text.empty()
    progress_bar.empty()

    st.metric("Total Return", f"{results['Total Return']:.2%}")
    st.metric("Annualized Return", f"{results['Annualized Return']:.2%}")
    st.metric("Sharpe Ratio", f"{results['Sharpe Ratio']:.2f}")
    st.metric("Max Drawdown", f"{results['Max Drawdown']:.2%}")

    st.subheader("📂 Sector Allocation")
    try:
        sector_allocation = pd.DataFrame.from_dict(results.get('Sector Allocation', {}), 
                                                 orient='index', 
                                                 columns=["Weight"])
        st.dataframe(sector_allocation)
        
        # Create and display interactive portfolio dashboard
        st.subheader("📊 Portfolio Analysis Dashboard")
        dashboard = create_portfolio_dashboard(
            weights_data=weights,
            sector_weights=results.get('Sector Allocation', {}),
            risk_metrics={
                'sharpe_ratio': results['Sharpe Ratio'],
                'total_return': results['Total Return'],
                'max_drawdown': results['Max Drawdown']
            }
        )
        st.plotly_chart(dashboard, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error displaying portfolio analysis: {str(e)}")
    
    # Step 6: Feedback Collection
    st.markdown("---")
    st.subheader("🔬 Proof of Concept & Feedback")
    
    # POC Disclaimer
    st.info("""
    👋 Thank you for trying out Axia! This is currently a Proof of Concept (POC) version, and we're actively working on improvements. 
    While you might encounter some bugs, your feedback is invaluable in helping us create a better product.
    """)
    
    # Feedback Form
    st.subheader("📝 Share Your Thoughts")
    user_experience = st.slider("How would you rate your experience?", 1, 5, 3)
    feedback_text = st.text_area("What could we improve?", height=100)
    
    # Feature Survey
    st.subheader("🚀 Future Features")
    st.write("We're considering adding the ability to create custom models and algorithms. How would you like to use this feature?")
    
    custom_model_preference = st.radio(
        "Would you be interested in creating custom investment models?",
        ['Yes, from scratch', 'Yes, using templates', 'No, not interested'],
        key='custom_model'
    )
    
    if custom_model_preference.startswith('Yes'):
        preferred_features = st.multiselect(
            "Which features would you find most useful?",
            [
                "Custom risk models",
                "Alternative data integration",
                "Machine learning templates",
                "Backtesting framework",
                "API integration",
                "Community model sharing"
            ]
        )
    
    # Contact Information
    st.subheader("📫 Stay Updated")
    st.write("Leave your email to receive updates about new features and improvements!")
    user_email = st.text_input("Email address")
    
    # Submit Button
    if st.button("Submit Feedback"):
        if user_email:
            # Here you would typically implement email sending logic
            st.success("Thank you for your feedback! We'll keep you updated on our progress.")
        else:
            st.warning("Please provide an email address to submit your feedback.")
