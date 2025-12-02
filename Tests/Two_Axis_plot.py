import plotly.graph_objects as go

window = 2000
df_sample = df.iloc[:window]

# Create figure
fig = go.Figure()

# --- Left y-axis (Price) ---
fig.add_trace(go.Scatter(
    x=df_sample.index, y=df_sample['Mid'],
    mode='lines', name='Mid-Price',
    line=dict(color='gray', width=1),
    yaxis='y1'
))
fig.add_trace(go.Scatter(
    x=df_sample.index, y=df_sample['KF_level_adapt'],
    mode='lines', name='KF Level',
    line=dict(color='royalblue', width=1.5),
    yaxis='y1'
))

# --- Right y-axis (Probabilities) ---
fig.add_trace(go.Scatter(
    x=df_sample.index, y=df_sample['Pmax'],
    mode='lines', name='Pmax (local max)',
    line=dict(color='red', width=1.5),
    yaxis='y2'
))
fig.add_trace(go.Scatter(
    x=df_sample.index, y=df_sample['Pmin'],
    mode='lines', name='Pmin (local min)',
    line=dict(color='green', width=1.5),
    yaxis='y2'
))

# --- Layout ---
fig.update_layout(
    title='Real-Time Extremum Probabilities (Single Chart)',
    xaxis=dict(title='Time'),
    # xaxis=dict(title='Time', automargin=True, rangeslider=dict(visible=False), constrain='domain'),
    yaxis=dict(title='Price', color='royalblue'),
    yaxis2=dict(
        title='Probability',
        color='red',
        overlaying='y',  # share the same x-axis
        side='right',
        range=[0, 1]
    ),
    legend=dict(x=0.02, y=0.98),
    height=700,
    width=1000,
    template='plotly_white',
    hovermode='x unified'
)

fig.show()
