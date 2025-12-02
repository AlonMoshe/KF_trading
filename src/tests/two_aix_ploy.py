import pandas as pd
import numpy as np
import plotly.graph_objects as go

start = '2025-11-17 09:30:00-05:00'
end   = '2025-11-17 15:59:55-05:00'
df_range = df_sig[(df_sig.index >= start) & (df_sig.index <= end)]

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_range.index,
    y=df_range['Mid'],
    mode='lines+markers',  # show both lines and dots
    name='Mid price',
    line=dict(color='darkorange', width=2),
    marker=dict(size=4, color='darkorange'),  # small dots
    yaxis='y1'
))
fig.add_trace(go.Scatter(
    x=df_range.index,
    y=df_range['KF_level_adapt'],
    mode='lines+markers',  # show both lines and dots
    name='KF_level_adapt',
    line=dict(color='green', width=2),
    marker=dict(size=4, color='green'),  # small dots
    yaxis='y1'
))

fig.add_trace(go.Scatter(
    x=df_range.index, y=1000*df_range['KF_slope_adapt'],
    mode='lines+markers', name='slope',
    line=dict(color='red', width=1.5),
    yaxis='y2'
))


fig.update_layout(
    title='Mid versus filter price output - Slope',
    xaxis=dict(title='Time'),
    # xaxis=dict(title='Time', automargin=True, rangeslider=dict(visible=False), constrain='domain'),
    yaxis=dict(title='Price', color='royalblue'),
    yaxis2=dict(
        title='Slope',
        color='red',
        overlaying='y',  # share the same x-axis
        side='right',
        range=[-50,50]
    ),
    legend=dict(x=0.02, y=0.98),
    height=700,
    width=1000,
    template='plotly_white',
    hovermode='x unified'
    
)



fig.show()