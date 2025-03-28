import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from plotly.subplots import make_subplots

def create_portfolio_dashboard(weights_data, sector_weights, risk_metrics):
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "pie"}, {"type": "bar"}],
               [{"type": "treemap"}, {"type": "indicator"}]],
        subplot_titles=("Sector Allocation", "Top 10 Holdings", "Asset Allocation Treemap", "Risk Metrics")
    )
    
    # 1. Sector Allocation (Pie Chart)
    sector_df = pd.DataFrame(list(sector_weights.items()), columns=['Sector', 'Weight'])
    sector_df = sector_df[sector_df['Sector'] != 'Unknown']
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
    
    # 2. Top 10 Holdings (Bar Chart)
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
    
    # 3. Asset Allocation Treemap
    # Create proper treemap data structure
    treemap_data = []
    
    # First add sectors as root nodes
    for sector in sector_weights:
        treemap_data.append({
            'id': sector,
            'parent': '',
            'value': sector_weights[sector]
        })
    
    # Then add assets as children of sectors
    for asset, weight in weights_data.items():
        # Use the sector from the sector_weights keys
        # This is a simplified approach that assigns all assets to the first sector
        # In a real implementation, we would need access to the sectors dictionary
        if len(sector_weights) > 0:
            asset_sector = list(sector_weights.keys())[0]
        else:
            asset_sector = 'Unknown'
        
        treemap_data.append({
            'id': asset,
            'parent': asset_sector,
            'value': weight
        })
    
    treemap_df = pd.DataFrame(treemap_data)
    
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
    fig.add_trace(
        go.Indicator(
            mode="number+gauge+delta",
            value=risk_metrics.get('sharpe_ratio', 0),
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
    
    # Add additional risk metrics as annotations
    fig.add_annotation(
        x=0.875, y=0.3,
        text=f"Expected Return: {risk_metrics.get('expected_return', 0):.2%}",
        showarrow=False,
        font=dict(size=14),
        xref="paper", yref="paper"
    )
    
    fig.add_annotation(
        x=0.875, y=0.2,
        text=f"Volatility: {risk_metrics.get('volatility', 0):.2%}",
        showarrow=False,
        font=dict(size=14),
        xref="paper", yref="paper"
    )
    
    fig.add_annotation(
        x=0.875, y=0.1,
        text=f"Max Drawdown: {risk_metrics.get('max_drawdown', 0):.2%}",
        showarrow=False,
        font=dict(size=14),
        xref="paper", yref="paper"
    )
    
    # Update layout with improved styling
    fig.update_layout(
        height=1000,
        showlegend=True,
        title_text="Portfolio Analysis Dashboard",
        template="plotly_dark",
        margin=dict(t=50, l=25, r=25, b=25),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Update axes for better readability
    fig.update_xaxes(title_text="Weight (%)", row=1, col=2)
    
    return fig