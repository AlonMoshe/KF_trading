import plotly.graph_objects as go



def plot_price_signals(
    df,
    price_cols=("KF_level_adapt",),
    secondary_cols=("KF_slope_adapt",),
    entry_long_col="entry_long",
    exit_long_col="exit_long",
    entry_short_col="entry_short",
    exit_short_col="exit_short",
    show_peaks=False,
    peak_min_col="slope_peak_min",
    peak_max_col="slope_peak_max",
    markersize=10,
    secondary_scale=1.0,
    title="Price and Signals",
    height=700,
    width=1200,
    template="plotly_white"
):
    """
    Plot price, secondary indicators, entries, exits, and peaks using Plotly.
    """

    fig = go.Figure()

    # --------------------------------------------------
    # Left Y-axis: Price series
    # --------------------------------------------------
    for col in price_cols:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[col],
            mode='lines',
            name=col,
            line=dict(width=2)
        ))

    # --------------------------------------------------
    # Right Y-axis: Secondary indicators (e.g., slope)
    # --------------------------------------------------
    for col in secondary_cols:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[col] * secondary_scale,
            mode='lines+markers',
            name=f"{col} (scaled)",
            line=dict(width=1.5, color='orange'),
            yaxis='y2'
        ))

    # Use the main price column as Y for markers
    base_col = price_cols[0]

    # --------------------------------------------------
    # Entry Long Marker (green triangle-up)
    # --------------------------------------------------
    if entry_long_col in df.columns:
        idx = df.index[df[entry_long_col] == True]
        fig.add_trace(go.Scatter(
            x=idx,
            y=df.loc[idx, base_col],
            mode='markers',
            marker=dict(symbol="triangle-up", size=markersize, color="green"),
            name="Long Entry"
        ))

    # --------------------------------------------------
    # Exit Long Marker (light green circle)
    # --------------------------------------------------
    if exit_long_col in df.columns:
        idx = df.index[df[exit_long_col] == True]
        fig.add_trace(go.Scatter(
            x=idx,
            y=df.loc[idx, base_col],
            mode='markers',
            marker=dict(symbol="circle", size=markersize, color="lightgreen"),
            name="Long Exit"
        ))

    # --------------------------------------------------
    # Entry Short Marker (red triangle-down)
    # --------------------------------------------------
    if entry_short_col in df.columns:
        idx = df.index[df[entry_short_col] == True]
        fig.add_trace(go.Scatter(
            x=idx,
            y=df.loc[idx, base_col],
            mode='markers',
            marker=dict(symbol="triangle-down", size=markersize, color="red"),
            name="Short Entry"
        ))

    # --------------------------------------------------
    # Exit Short Marker (pink circle)
    # --------------------------------------------------
    if exit_short_col in df.columns:
        idx = df.index[df[exit_short_col] == True]
        fig.add_trace(go.Scatter(
            x=idx,
            y=df.loc[idx, base_col],
            mode='markers',
            marker=dict(symbol="circle", size=markersize, color="pink"),
            name="Short Exit"
        ))
        
    # --------------------------------------------------
    # Add numeric labels next to entries/exits
    # --------------------------------------------------

    # Long trades
    if "trade_id_long" in df.columns:
        idx = df.index[df["trade_id_long"].notna()]
        for t in idx:
            fig.add_annotation(
                x=t,
                y=df.loc[t, base_col],
                text=str(int(df.loc[t, "trade_id_long"])),
                showarrow=False,
                yshift=15,
                font=dict(size=10, color="green")
            )

    # Short trades
    if "trade_id_short" in df.columns:
        idx = df.index[df["trade_id_short"].notna()]
        for t in idx:
            fig.add_annotation(
                x=t,
                y=df.loc[t, base_col],
                text=str(int(df.loc[t, "trade_id_short"])),
                showarrow=False,
                yshift=-15,
                font=dict(size=10, color="red")
            )


    # --------------------------------------------------
    # Optional: slope peaks (for diagnostics)
    # --------------------------------------------------
    if show_peaks:
        # Min peaks
        if peak_min_col in df.columns:
            idx = df.index[df[peak_min_col] == True]
            fig.add_trace(go.Scatter(
                x=idx,
                y=df.loc[idx, base_col],
                mode="markers",
                marker=dict(symbol="circle", size=markersize/2, color="darkgreen"),
                name="Slope Trough"
            ))

        # Max peaks
        if peak_max_col in df.columns:
            idx = df.index[df[peak_max_col] == True]
            fig.add_trace(go.Scatter(
                x=idx,
                y=df.loc[idx, base_col],
                mode="markers",
                marker=dict(symbol="circle", size=markersize/2, color="darkred"),
                name="Slope Peak"
            ))

    # --------------------------------------------------
    # Layout
    # --------------------------------------------------
    fig.update_layout(
        title=title,
        height=height,
        width=width,
        template=template,
        hovermode="x unified",
        yaxis=dict(title="Price"),
        yaxis2=dict(
            title="Secondary",
            overlaying='y',
            side='right',
            showgrid=False
        ),
        legend=dict(x=0.01, y=0.99)
    )

    fig.show()
