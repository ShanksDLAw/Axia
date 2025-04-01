import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import logging
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Optional, Union

def create_portfolio_dashboard(weights_data: Dict[str, float], sector_weights: Dict[str, float], risk_metrics: Dict[str, Any], sectors_map: Optional[Dict[str, str]] = None) -> go.Figure:
    # Validate and clean input data
    try:
        if not isinstance(weights_data, dict) or not weights_data:
            raise ValueError("Invalid weights data provided")
            
        if not isinstance(sector_weights, dict) or not sector_weights:
            sector_weights = {'Uncategorized': 1.0}
            
        if not isinstance(risk_metrics, dict):
            raise ValueError("Invalid risk metrics data provided")
    except Exception as e:
        logging.error(f"Error validating input data: {str(e)}")
        weights_data = {}
        sector_weights = {'Uncategorized': 1.0}
        risk_metrics = {}
    
    # Normalize sector weights if they don't sum to 1 with improved validation
    try:
        # First filter out invalid values and handle NaN/zero weights
        valid_sector_weights = {}
        for sector, weight in sector_weights.items():
            if isinstance(weight, (int, float)) and not np.isnan(weight) and weight > 0:
                valid_sector_weights[sector] = float(weight)
            else:
                logging.warning(f"Invalid or zero sector weight for {sector}: {weight}")
        
        # If no valid weights, use default with warning
        if not valid_sector_weights:
            logging.warning("No valid sector weights found. Using default equal sector allocation.")
            sector_weights = {'Uncategorized': 1.0}
        else:
            # Normalize the valid weights with better precision handling
            total_sector_weight = sum(valid_sector_weights.values())
            if total_sector_weight > 0:
                if not np.isclose(total_sector_weight, 1.0, rtol=1e-5, atol=1e-8):
                    sector_weights = {k: v/total_sector_weight for k, v in valid_sector_weights.items()}
                    logging.info(f"Normalized sector weights from sum {total_sector_weight:.6f} to 1.0")
                else:
                    sector_weights = valid_sector_weights
            else:
                logging.warning("Total sector weight is zero or negative. Using default allocation.")
                sector_weights = {'Uncategorized': 1.0}
    except Exception as e:
        logging.error(f"Error normalizing sector weights: {str(e)}")
        sector_weights = {'Uncategorized': 1.0}
    
    # Create figure with optimized layout for medium risk profile
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "pie", "rowspan": 1}, {"type": "bar"}],
               [{"type": "treemap"}, {"type": "indicator"}]],
        subplot_titles=("Sector Allocation", "Top 10 Holdings", "Asset Allocation Treemap", "Risk Profile Metrics"),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # 1. Sector Allocation (Pie Chart) with improved validation
    # Filter out invalid sectors and weights
    valid_sectors = {}
    for sector, weight in sector_weights.items():
        if not isinstance(weight, (int, float)) or np.isnan(weight) or weight <= 0:
            continue
        valid_sectors[sector] = weight
            
    # Create dataframe from valid sectors
    sector_df = pd.DataFrame(list(valid_sectors.items()), columns=['Sector', 'Weight'])
    sector_df = sector_df[sector_df['Sector'] != 'Unknown']
    if not sector_df.empty:
        fig.add_trace(
            go.Pie(
                labels=sector_df['Sector'], 
                values=sector_df['Weight'], 
                hole=0.4,
                textinfo='label+percent',
                marker=dict(colors=px.colors.qualitative.Plotly)
            ),
            row=1, col=1
        )
    else:
        # Add a placeholder if no sectors
        fig.add_trace(
            go.Pie(
                labels=['No Sector Data'], 
                values=[1], 
                hole=0.4,
                textinfo='label',
                marker=dict(colors=['gray'])
            ),
            row=1, col=1
        )
    
    # 2. Top 10 Holdings (Bar Chart)
    if weights_data:
        holdings_df = pd.DataFrame(list(weights_data.items()), columns=['Asset', 'Weight'])
        holdings_df = holdings_df.sort_values('Weight', ascending=False).head(10)
        holdings_df['Weight_Percent'] = holdings_df['Weight'] * 100  # Convert to percentage for display
        
        fig.add_trace(
            go.Bar(
                x=holdings_df['Weight_Percent'],
                y=holdings_df['Asset'],
                orientation='h',
                text=holdings_df['Weight_Percent'].apply(lambda x: f'{x:.2f}%'),
                textposition='auto',
                marker=dict(color=px.colors.sequential.Blues_r)
            ),
            row=1, col=2
        )
    else:
        # Add placeholder if no holdings data
        fig.add_annotation(
            x=0.75, y=0.75,
            text="No holdings data available",
            showarrow=False,
            font=dict(size=14),
            xref="paper", yref="paper"
        )
    
    # 3. Asset Allocation Treemap
    # Create proper treemap data structure with improved sector mapping
    treemap_data = []
    
    # First add sectors as root nodes with validation
    valid_sectors = {}
    for sector, weight in sector_weights.items():
        if not isinstance(weight, (int, float)) or np.isnan(weight) or weight <= 0:
            continue
        valid_sectors[sector] = weight
        treemap_data.append({
            'id': sector,
            'parent': '',
            'value': weight
        })
    
    # Handle case with no valid sectors
    if not valid_sectors:
        valid_sectors = {'Uncategorized': 1.0}
        treemap_data.append({
            'id': 'Uncategorized',
            'parent': '',
            'value': 1.0
        })
    
    # Then add assets as children of sectors with improved mapping
    for asset, weight in weights_data.items():
        # Skip invalid weights
        if not isinstance(weight, (int, float)) or np.isnan(weight) or weight <= 0:
            continue
            
        # Determine the sector for this asset with better fallback handling
        asset_sector = 'Uncategorized'
        
        # Try to get sector from sectors_map if provided
        if sectors_map and asset in sectors_map:
            mapped_sector = sectors_map[asset]
            # Only use the mapped sector if it exists in our valid sectors
            if mapped_sector in valid_sectors:
                asset_sector = mapped_sector
        
        # Add the asset to the treemap
        treemap_data.append({
            'id': asset,
            'parent': asset_sector,
            'value': weight
        })
    
    treemap_df = pd.DataFrame(treemap_data)
    
    if not treemap_df.empty:
        fig.add_trace(
            go.Treemap(
                ids=treemap_df['id'],
                parents=treemap_df['parent'],
                values=treemap_df['value'],
                branchvalues='total',
                textinfo='label+value+percent root',
                marker=dict(colorscale=px.colors.sequential.Viridis)
            ),
            row=2, col=1
        )
    
    # 4. Risk Metrics Indicators with improved visualization
    # Safely get risk metrics with proper defaults
    sharpe_ratio = risk_metrics.get('sharpe_ratio', 0)
    expected_return = risk_metrics.get('expected_return', 0)
    volatility = risk_metrics.get('volatility', 0)
    max_drawdown = risk_metrics.get('max_drawdown', 0)
    
    # Ensure numeric values
    try:
        sharpe_ratio = float(sharpe_ratio)
        expected_return = float(expected_return)
        volatility = float(volatility)
        max_drawdown = float(max_drawdown)
    except (ValueError, TypeError):
        logging.warning("Non-numeric risk metrics detected, using defaults")
        sharpe_ratio = 0.0
        expected_return = 0.0
        volatility = 0.0
        max_drawdown = 0.0
    
    fig.add_trace(
        go.Indicator(
            mode="number+gauge+delta",
            value=sharpe_ratio,
            title={'text': "Sharpe Ratio"},
            gauge={
                'axis': {'range': [None, 3]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 1], 'color': "lightgray"},
                    {'range': [1, 2], 'color': "gray"},
                    {'range': [2, 3], 'color': "darkgray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 1
                }
            },
            domain={'x': [0, 1], 'y': [0, 1]}
        ),
        row=2, col=2
    )
    
    # Format percentage values safely
    def format_percentage(value):
        try:
            return f"{float(value):.2%}"
        except (ValueError, TypeError):
            return "0.00%"
    
    # Add additional risk metrics as annotations
    fig.add_annotation(
        x=0.875, y=0.3,
        text=f"Expected Return: {format_percentage(expected_return)}",
        showarrow=False,
        font=dict(size=14),
        xref="paper", yref="paper"
    )
    
    fig.add_annotation(
        x=0.875, y=0.2,
        text=f"Volatility: {format_percentage(volatility)}",
        showarrow=False,
        font=dict(size=14),
        xref="paper", yref="paper"
    )
    
    fig.add_annotation(
        x=0.875, y=0.1,
        text=f"Max Drawdown: {format_percentage(max_drawdown)}",
        showarrow=False,
        font=dict(size=14),
        xref="paper", yref="paper"
    )
    
    # Update layout with improved styling and spacing
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="Portfolio Analysis Dashboard",
        title_x=0.5,
        template="plotly_dark",
        margin=dict(t=80, l=50, r=50, b=50, pad=4),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=10)
        ),
        uniformtext=dict(minsize=10, mode='hide'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # Update axes for better readability
    fig.update_xaxes(title_text="Weight (%)", row=1, col=2, showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    
    # Adjust pie chart settings for better visibility
    fig.update_traces(
        textposition='outside',
        textinfo='label+percent',
        hoverinfo='label+percent',
        row=1, col=1
    )
    
    # Enhance bar chart appearance
    fig.update_traces(
        texttemplate='%{text:.2f}%',
        textposition='inside',
        hovertemplate='%{y}: %{x:.2f}%<extra></extra>',
        row=1, col=2
    )
    
    return fig