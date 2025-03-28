# Axia Portfolio Analysis Tool

Axia is a sophisticated portfolio analysis and optimization tool that helps investors make data-driven decisions about their investment portfolios. The tool combines modern portfolio theory with advanced visualization techniques to provide comprehensive insights into portfolio performance and risk metrics.

## Features

- **Portfolio Optimization**: Implements modern portfolio theory to optimize asset allocation
- **Interactive Visualizations**: Provides detailed portfolio analysis through interactive dashboards
- **Risk Analysis**: Calculates key risk metrics including Sharpe Ratio, volatility, and maximum drawdown
- **Sector Analysis**: Visualizes sector allocation and provides detailed breakdown of portfolio composition

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Dependencies

- yfinance: For fetching financial data
- pandas: For data manipulation and analysis
- numpy: For numerical computations
- streamlit: For creating the web interface
- plotly: For interactive visualizations

## Usage

To run the application:

```bash
streamlit run app.py
```

This will start the web interface where you can:
1. Input your portfolio composition
2. View interactive visualizations
3. Analyze risk metrics
4. Optimize portfolio allocation

## Project Structure

- `app.py`: Main application entry point
- `asset_loader.py`: Handles data loading and processing
- `market_analyser.py`: Performs market analysis
- `optimiser.py`: Implements portfolio optimization algorithms
- `visualization.py`: Creates interactive dashboards and plots
- `backtester.py`: Implements portfolio backtesting functionality