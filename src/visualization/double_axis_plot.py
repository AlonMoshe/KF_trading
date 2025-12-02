import plotly.graph_objects as go

def plot_dual_axis(df, left_cols, right_cols, title="Dual Axis Plot"):
    """
    Plot multiple dataframe columns:
    - left_cols  -> left y-axis
    - right_cols -> right y-axis (raw values, no normalization)
    """

    fig = go.Figure()

    # -----------------------------------------------------------
    # LEFT AXIS PLOTS
    # -----------------------------------------------------------
    for col in left_cols:
        if col not in df.columns:
            print(f"Column '{col}' not found in DataFrame.")
            continue

        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[col],
            mode="lines+markers",
            name=col,
            line=dict(width=2),
            yaxis="y"
        ))

    # -----------------------------------------------------------
    # RIGHT AXIS PLOTS
    # -----------------------------------------------------------
    for col in right_cols:
        if col not in df.columns:
            print(f"Column '{col}' not found in DataFrame.")
            continue

        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[col],
            mode="lines+markers",
            name=col,
            line=dict(width=2),
            yaxis="y2"
        ))

    # -----------------------------------------------------------
    # LAYOUT
    # -----------------------------------------------------------
    fig.update_layout(
        title=title,
        height=800,
        width=1200,
        template="plotly_white",
        hovermode="x unified",

        yaxis=dict(
            title="Left Axis",
            side="left"
        ),

        yaxis2=dict(
            title="Right Axis",
            side="right",
            overlaying="y",
            showgrid=False
        ),

        legend=dict(x=0.01, y=0.99)
    )

    fig.show()
