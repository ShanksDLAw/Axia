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
    risk_appetite = 'Moderate'  # Default value
    if risk_level:
        if risk_level == 'Low':
            risk_appetite = 'Conservative'
        elif risk_level == 'High':
            risk_appetite = 'Aggressive'
        else:
            risk_appetite = 'Moderate'
    
    # Set default constraints
    constraints = {
        'max_position': 0.1,
        'max_sector': 0.3,
        'min_position': 0.01
    }
    
    # Call optimize with all required parameters
    try:
        weights, portfolio_metrics = optimizer.optimize(regime, risk_appetite, constraints)
        if weights is None or portfolio_metrics is None:
            # Fallback to equal weight if optimization fails
            total_assets = len(price_data.columns)
            weights = {symbol: 1.0/total_assets for symbol in price_data.columns}
            portfolio_metrics = {
                'expected_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'num_assets': total_assets,
                'total_weight': 1.0,
                'warning': 'Using equal weight fallback due to optimization failure'
            }
    except Exception as e:
        # Fallback to equal weight if exception occurs
        total_assets = len(price_data.columns)
        weights = {symbol: 1.0/total_assets for symbol in price_data.columns}
        portfolio_metrics = {
            'expected_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'num_assets': total_assets,
            'total_weight': 1.0,
            'warning': f'Using equal weight fallback due to: {str(e)}'
        }
        
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
        # Calculate actual number of assets with non-zero weights
        active_assets = sum(1 for w in weights.values() if w > 0.0001)
        st.metric("Diversification", f"{active_assets} assets")
    
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
        
        # Validate sector allocation data before visualization
        sector_allocation_data = results.get('Sector Allocation', {})
        if not sector_allocation_data or not isinstance(sector_allocation_data, dict):
            sector_allocation_data = {'Uncategorized': 1.0}
            st.warning("No valid sector allocation data available. Using default categorization.")
            
        # Ensure all sector weights are valid numbers
        validated_sectors = {}
        for sector, weight in sector_allocation_data.items():
            try:
                weight_val = float(weight)
                if not np.isnan(weight_val) and weight_val > 0:
                    validated_sectors[sector] = weight_val
            except (ValueError, TypeError):
                continue
                
        # If no valid sectors remain, use a default
        if not validated_sectors:
            validated_sectors = {'Uncategorized': 1.0}
            
        dashboard = create_portfolio_dashboard(
            weights_data=weights,
            sector_weights=validated_sectors,
            risk_metrics={
                'sharpe_ratio': results['Sharpe Ratio'],
                'total_return': results['Total Return'],
                'max_drawdown': results['Max Drawdown'],
                'volatility': portfolio_metrics.get('volatility', 0.0)
            },
            sectors_map=sectors
        )
        st.plotly_chart(dashboard, use_container_width=True, theme="streamlit")
        
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
     
    # Initialize session state for form data persistence
    if 'feedback_submitted' not in st.session_state:
        st.session_state.feedback_submitted = False
    if 'form_data' not in st.session_state:
        st.session_state.form_data = {
            'user_experience': 3,
            'feedback_text': '',
            'custom_model_preference': 'No, not interested',
            'preferred_features': [],
            'user_email': ''
        }
    
    # Function to save feedback
    def save_feedback(feedback_data):
        try:
            import json
            import os
            import smtplib
            from datetime import datetime
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            # Create feedback directory if it doesn't exist
            feedback_dir = os.path.join(os.path.dirname(__file__), 'feedback')
            os.makedirs(feedback_dir, exist_ok=True)
            
            # Add timestamp to feedback data
            feedback_data['timestamp'] = datetime.now().isoformat()
            
            # Generate unique filename
            filename = f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(feedback_dir, filename)
            
            # Save feedback to file
            with open(filepath, 'w') as f:
                json.dump(feedback_data, f, indent=4)
            
            # Send email notification
            try:
                # Email configuration
                sender_email = "your-app-email@gmail.com"  # Replace with your app's email
                receiver_email = "animicart@gmail.com"
                app_password = os.environ.get('EMAIL_APP_PASSWORD')  # Set this in your environment variables
                
                if not app_password:
                    logging.warning("Email app password not found in environment variables")
                    return True  # Continue with success even if email fails
                
                # Create email message
                msg = MIMEMultipart()
                msg['From'] = sender_email
                msg['To'] = receiver_email
                msg['Subject'] = f"New Axia Feedback - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                
                # Format feedback content
                email_content = f"""New feedback received from Axia:
                
User Experience Rating: {feedback_data['user_experience']}/5
Feedback Text: {feedback_data['feedback_text']}
Custom Model Preference: {feedback_data['custom_model_preference']}
Preferred Features: {', '.join(feedback_data['preferred_features']) if feedback_data['preferred_features'] else 'None'}
User Email: {feedback_data['user_email']}
                """
                
                msg.attach(MIMEText(email_content, 'plain'))
                
                # Send email
                with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                    server.login(sender_email, app_password)
                    server.send_message(msg)
                    
                logging.info("Feedback email sent successfully")
            except Exception as email_error:
                logging.error(f"Failed to send feedback email: {str(email_error)}")
                # Continue with success even if email fails
                
            return True
        except Exception as e:
            st.error(f"Error saving feedback: {str(e)}")
            return False

    # Feedback Form with error handling
    try:
        with st.form(key='feedback_form'):
            st.subheader("📝 Share Your Thoughts")
            
            # User Experience Rating
            st.session_state.form_data['user_experience'] = st.slider(
                "How would you rate your experience?",
                min_value=1,
                max_value=5,
                value=st.session_state.form_data['user_experience'],
                help="Slide to rate your experience from 1 to 5"
            )
            
            # Feedback Text
            st.session_state.form_data['feedback_text'] = st.text_area(
                "What could we improve?",
                value=st.session_state.form_data['feedback_text'],
                height=100,
                help="Share your thoughts on how we can improve"
            )
            
            # Feature Survey
            st.subheader("🚀 Future Features")
            st.write("We're considering adding the ability to create custom models and algorithms. How would you like to use this feature?")
            
            st.session_state.form_data['custom_model_preference'] = st.radio(
                "Would you be interested in creating custom investment models?",
                ['Yes, from scratch', 'Yes, using templates', 'No, not interested'],
                index=['Yes, from scratch', 'Yes, using templates', 'No, not interested'].index(
                    st.session_state.form_data['custom_model_preference']
                )
            )
            
            # Show feature selection only if interested
            if st.session_state.form_data['custom_model_preference'].startswith('Yes'):
                st.session_state.form_data['preferred_features'] = st.multiselect(
                    "Which features would you find most useful?",
                    [
                        "Custom risk models",
                        "Alternative data integration",
                        "Machine learning templates",
                        "Backtesting framework",
                        "API integration",
                        "Community model sharing"
                    ],
                    default=st.session_state.form_data['preferred_features']
                )
            
            # Contact Information
            st.subheader("📫 Stay Updated")
            st.write("Leave your email to receive updates about new features and improvements!")
            st.session_state.form_data['user_email'] = st.text_input(
                "Email address",
                value=st.session_state.form_data['user_email'],
                help="Enter your email to receive updates"
            )
            
            # Submit Button
            submit_button = st.form_submit_button("Submit Feedback")
            
            if submit_button:
                try:
                    # Validate email format
                    if not st.session_state.form_data['user_email'] or '@' not in st.session_state.form_data['user_email']:
                        st.error("Please provide a valid email address.")
                    else:
                        # Save feedback data
                        feedback_saved = save_feedback(st.session_state.form_data)
                        
                        if feedback_saved:
                            st.session_state.feedback_submitted = True
                            st.success("Thank you for your feedback! We'll keep you updated on our progress.")
                            
                            # Clear form data after successful submission
                            st.session_state.form_data = {
                                'user_experience': 3,
                                'feedback_text': '',
                                'custom_model_preference': 'No, not interested',
                                'preferred_features': [],
                                'user_email': ''
                            }
                        else:
                            st.error("Failed to save feedback. Please try again.")
                            st.session_state.feedback_submitted = False
                except Exception as e:
                    st.error(f"An error occurred while submitting feedback. Please try again.")
                    logging.error(f"Feedback submission error: {str(e)}")
                    
    except Exception as e:
        st.error("An error occurred while rendering the feedback form. Please refresh the page.")
        logging.error(f"Feedback form error: {str(e)}")
