import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_custom(df):

    # ---------------------------
    # Normalize two series for right axis
    # ---------------------------
    def normalize(s):
        return (s - s.min()) / (s.max() - s.min())

    slope_norm = normalize(df["KF_slope_adapt"])
    curv_norm  = normalize(df["KF_curv_var_adapt"])

    fig = go.Figure()

    # -----------------------------------------------------------
    # LEFT AXIS: Mid + KF_level_adapt
    # -----------------------------------------------------------
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["Mid"],
        mode='lines',
        name='Mid',
        line=dict(width=2)
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["KF_level_adapt"],
        mode='lines',
        name='KF_level_adapt',
        line=dict(width=2)
    ))

    # -----------------------------------------------------------
    # Markers on Mid where slope has peaks or troughs
    # -----------------------------------------------------------
    # Peaks
    idx_peak = df.index[df["slope_peak_max"] == True]
    fig.add_trace(go.Scatter(
        x=idx_peak,
        y=df.loc[idx_peak, "Mid"],
        mode="markers",
        marker=dict(symbol="circle", color="red", size=8),
        name="Slope Peak"
    ))

    # Troughs
    idx_trough = df.index[df["slope_peak_min"] == True]
    fig.add_trace(go.Scatter(
        x=idx_trough,
        y=df.loc[idx_trough, "Mid"],
        mode="markers",
        marker=dict(symbol="circle", color="green", size=8),
        name="Slope Trough"
    ))

    # -----------------------------------------------------------
    # RIGHT AXIS: Normalized slope + normalized curvature
    # -----------------------------------------------------------
    fig.add_trace(go.Scatter(
        x=df.index,
        y=slope_norm,
        mode='lines+markers',
        name='KF_slope_adapt (norm)',
        # line=dict(width=1.5, dash='dot'),
        line=dict(width=1.5),
        yaxis='y2'
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=curv_norm,
        mode='lines+markers',
        name='KF_curv_var_adapt (norm)',
        # line=dict(width=1.5, dash='dash'),
        line=dict(width=1.5),
        yaxis='y2'
    ))

    # -----------------------------------------------------------
    # Layout
    # -----------------------------------------------------------
    fig.update_layout(
        title="Mid, Level, Slope, Curvature",
        height=800,
        width=1200,
        template="plotly_white",
        hovermode="x unified",

        yaxis=dict(
            title="Price (Mid / KF Level)",
            side="left"
        ),

        yaxis2=dict(
            title="Normalized Slope / Curvature",
            overlaying='y',
            side='right',
            showgrid=False
        ),

        legend=dict(x=0.01, y=0.99)
    )

    fig.show()
