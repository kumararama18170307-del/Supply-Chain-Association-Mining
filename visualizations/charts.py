"""
Plotly Visualizations Module
Renders interactive analytical charts adhering to the Dark Navy / Blue theme.
All charts are constructed strictly from empirical dataset and mining outputs.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


THEME_COLORS = {
    "bg": "#0B132B",
    "card_bg": "#1C2541",
    "accent_blue": "#3A86FF",
    "accent_cyan": "#00B4D8",
    "accent_teal": "#48CAE4",
    "accent_green": "#06D6A0",
    "accent_yellow": "#FFD166",
    "accent_coral": "#EF476F",
    "text_primary": "#F8FAFC",
    "text_muted": "#94A3B8",
    "grid": "#233554"
}


def apply_theme(fig: go.Figure, **kwargs) -> go.Figure:
    """Safely applies dark navy theme defaults without duplicate keyword arguments."""
    defaults = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=THEME_COLORS["text_primary"], family="Segoe UI, Inter, sans-serif"),
        margin=dict(l=40, r=40, t=50, b=40)
    )
    defaults.update(kwargs)
    fig.update_layout(**defaults)
    fig.update_xaxes(gridcolor=THEME_COLORS["grid"], zerolinecolor=THEME_COLORS["grid"])
    fig.update_yaxes(gridcolor=THEME_COLORS["grid"], zerolinecolor=THEME_COLORS["grid"])
    return fig


def plot_top_products(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """
    Renders horizontal bar chart of top purchased products by total quantity.
    """
    if df.empty or "Product" not in df.columns:
        fig = go.Figure()
        return apply_theme(fig, title="Top Procured Products (No Data)")

    top_df = df.groupby("Product").agg(
        Total_Quantity=("Quantity", "sum"),
        Order_Count=("Order_ID", "nunique")
    ).reset_index().sort_values(by="Total_Quantity", ascending=True).tail(top_n)

    fig = px.bar(
        top_df,
        x="Total_Quantity",
        y="Product",
        orientation="h",
        title=f"Top {top_n} Procured Products by Volume",
        color="Total_Quantity",
        color_continuous_scale=["#1C2541", "#3A86FF", "#00B4D8"],
        text="Total_Quantity"
    )
    apply_theme(fig)
    fig.update_traces(textposition="outside", marker_line_color=THEME_COLORS["accent_cyan"], marker_line_width=1)
    fig.update_coloraxes(showscale=False)
    return fig


def plot_frequent_itemsets(freq_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """
    Renders bar chart of top frequent itemsets and their support.
    """
    if freq_df.empty:
        fig = go.Figure()
        return apply_theme(fig, title="Frequent Itemsets (None Found)")

    display_df = freq_df.sort_values(by="support", ascending=False).head(top_n).copy()
    display_df["support_pct"] = display_df["support"] * 100

    fig = px.bar(
        display_df,
        x="support_pct",
        y="itemset_str",
        orientation="h",
        title=f"Top {top_n} Frequent Itemsets (Support %)",
        color="length",
        color_continuous_scale=["#00B4D8", "#3A86FF", "#7209B7"],
        labels={"support_pct": "Support (% of Transactions)", "itemset_str": "Frequent Itemset", "length": "Item Count"}
    )
    apply_theme(fig)
    fig.update_yaxes(autorange="reversed")
    fig.update_traces(texttemplate="%{x:.1f}%", textposition="outside")
    return fig


def plot_itemset_size_distribution(freq_df: pd.DataFrame) -> go.Figure:
    """
    Displays distribution of itemset sizes (1-itemsets, 2-itemsets, etc.).
    """
    if freq_df.empty or "length" not in freq_df.columns:
        fig = go.Figure()
        return apply_theme(fig, title="Frequent Itemset Sizes (None Found)")

    size_counts = freq_df["length"].value_counts().reset_index()
    size_counts.columns = ["Itemset Size (k)", "Count"]
    size_counts = size_counts.sort_values(by="Itemset Size (k)")

    fig = px.bar(
        size_counts,
        x="Itemset Size (k)",
        y="Count",
        title="Distribution of Frequent Itemset Sizes (k-itemsets)",
        text="Count",
        color="Count",
        color_continuous_scale=["#3A86FF", "#06D6A0"]
    )
    apply_theme(fig)
    fig.update_traces(textposition="outside")
    fig.update_coloraxes(showscale=False)
    return fig


def plot_support_vs_confidence(rules_df: pd.DataFrame) -> go.Figure:
    """
    Scatter plot of Support vs Confidence, sized and colored by Lift.
    """
    if rules_df.empty:
        fig = go.Figure()
        return apply_theme(fig, title="Association Rules (No Rules Match Thresholds)")

    display_df = rules_df.copy()
    display_df["support_pct"] = display_df["support"] * 100
    display_df["confidence_pct"] = display_df["confidence"] * 100

    if "rule_repr" not in display_df.columns:
        if "antecedents_str" in display_df.columns and "consequents_str" in display_df.columns:
            display_df["rule_repr"] = display_df.apply(
                lambda r: f"{{{r['antecedents_str']}}} -> {{{r['consequents_str']}}}", axis=1
            )
        else:
            display_df["rule_repr"] = "Rule"

    fig = px.scatter(
        display_df,
        x="support_pct",
        y="confidence_pct",
        size="lift",
        color="lift",
        hover_name="rule_repr",
        hover_data={"support_pct": ":.2f%", "confidence_pct": ":.2f%", "lift": ":.2f"},
        title="Association Rules: Support vs Confidence (Color & Size = Lift)",
        color_continuous_scale=["#00B4D8", "#3A86FF", "#FFD166", "#EF476F"],
        labels={"support_pct": "Support (%)", "confidence_pct": "Confidence (%)", "lift": "Lift"}
    )
    apply_theme(fig)
    fig.update_traces(marker=dict(line=dict(width=1, color="white"), opacity=0.85))
    return fig


def plot_lift_comparison(rules_df: pd.DataFrame, top_n: int = 12) -> go.Figure:
    """
    Horizontal bar chart showing top rules sorted by Lift.
    """
    if rules_df.empty:
        fig = go.Figure()
        return apply_theme(fig, title="Rules by Lift (None Found)")

    top_rules = rules_df.sort_values(by="lift", ascending=False).head(top_n).copy()
    
    fig = px.bar(
        top_rules,
        x="lift",
        y="rule_repr",
        orientation="h",
        title=f"Top {top_n} Rules by Association Strength (Lift)",
        color="lift",
        color_continuous_scale=["#3A86FF", "#06D6A0", "#FFD166"],
        labels={"lift": "Lift Factor", "rule_repr": "Association Rule"}
    )
    apply_theme(fig)
    fig.update_yaxes(autorange="reversed")
    fig.update_traces(texttemplate="%{x:.2f}x", textposition="outside")
    fig.update_coloraxes(showscale=False)
    fig.add_vline(x=1.0, line_dash="dash", line_color=THEME_COLORS["accent_coral"], annotation_text="Independence (Lift=1.0)")
    return fig


def plot_rules_network(rules_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """
    Renders an interactive product association network graph.
    Nodes = Products; Edges = Rules weighted by Lift.
    """
    if rules_df.empty:
        fig = go.Figure()
        return apply_theme(fig, title="Product Association Network (No Rules Available)")

    top_rules = rules_df.sort_values(by="lift", ascending=False).head(top_n)

    nodes_set = set()
    edges = []

    for _, row in top_rules.iterrows():
        ant_list = [x.strip() for x in str(row.get("antecedents_str", "")).split(",") if x.strip()]
        con_list = [x.strip() for x in str(row.get("consequents_str", "")).split(",") if x.strip()]
        lift = float(row.get("lift", 1.0))
        conf = float(row.get("confidence", 0.5))

        for a in ant_list:
            nodes_set.add(a)
            for c in con_list:
                nodes_set.add(c)
                edges.append((a, c, lift, conf))

    nodes = list(nodes_set)
    if not nodes:
        fig = go.Figure()
        return apply_theme(fig, title="Product Association Network (No Nodes)")

    num_nodes = len(nodes)
    angles = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)
    pos = {node: (np.cos(angle), np.sin(angle)) for node, angle in zip(nodes, angles)}

    edge_x = []
    edge_y = []

    for a, c, lift, conf in edges:
        x0, y0 = pos[a]
        x1, y1 = pos[c]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color=THEME_COLORS["accent_blue"]),
        hoverinfo="none",
        mode="lines"
    )

    node_x = [pos[n][0] for n in nodes]
    node_y = [pos[n][1] for n in nodes]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=[n if len(n) <= 20 else n[:18] + ".." for n in nodes],
        textposition="top center",
        hovertext=nodes,
        hoverinfo="text",
        marker=dict(
            showscale=False,
            color=THEME_COLORS["accent_cyan"],
            size=18,
            line=dict(width=2, color="white")
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    apply_theme(
        fig,
        title="Product Association Network Graph (Rules Interconnection)",
        showlegend=False,
        hovermode="closest"
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
    return fig


def plot_warehouse_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Renders stacked bar chart of product orders across warehouses.
    """
    if df.empty or "Warehouse" not in df.columns:
        fig = go.Figure()
        return apply_theme(fig, title="Warehouse Inventory Flow (No Data)")

    wh_df = df.groupby(["Warehouse", "Product"])["Quantity"].sum().reset_index()
    top_products = df.groupby("Product")["Quantity"].sum().nlargest(8).index
    wh_df = wh_df[wh_df["Product"].isin(top_products)]

    fig = px.bar(
        wh_df,
        x="Warehouse",
        y="Quantity",
        color="Product",
        title="Warehouse Inventory Flow by Top SKUs",
        color_discrete_sequence=["#3A86FF", "#00B4D8", "#06D6A0", "#FFD166", "#EF476F", "#8338EC", "#FB5607", "#38B000"]
    )
    apply_theme(fig, barmode="stack")
    return fig


def plot_supplier_procurement(df: pd.DataFrame) -> go.Figure:
    """
    Renders procurement volume by supplier.
    """
    if df.empty or "Supplier" not in df.columns:
        fig = go.Figure()
        return apply_theme(fig, title="Procurement Volume by Supplier (No Data)")

    supp_df = df.groupby("Supplier").agg(
        Total_Units=("Quantity", "sum"),
        Unique_SKUs=("Product", "nunique")
    ).reset_index().sort_values(by="Total_Units", ascending=True)

    fig = px.bar(
        supp_df,
        x="Total_Units",
        y="Supplier",
        orientation="h",
        title="Procurement Volume by Supplier",
        color="Unique_SKUs",
        color_continuous_scale=["#1C2541", "#3A86FF", "#06D6A0"],
        labels={"Total_Units": "Total Units Procured", "Unique_SKUs": "Catalog SKUs"}
    )
    apply_theme(fig)
    fig.update_traces(texttemplate="%{x:,}", textposition="outside")
    return fig


def plot_runtime_comparison(metrics_apriori: Dict[str, Any], metrics_spark: Dict[str, Any]) -> go.Figure:
    """
    Renders side-by-side benchmark bar chart comparing Apriori vs FP-Growth runtime.
    """
    t_ap = float(metrics_apriori.get("execution_time_sec", 0.0) or 0.0)
    t_sp = float(metrics_spark.get("execution_time_sec", 0.0) or 0.0)

    times = [
        {"Algorithm": "Apriori (Level-wise Candidate Gen)", "Time (Seconds)": t_ap},
        {"Algorithm": "FP-Growth (FP-Tree Pattern Growth)", "Time (Seconds)": t_sp}
    ]
    comp_df = pd.DataFrame(times)

    fig = px.bar(
        comp_df,
        x="Algorithm",
        y="Time (Seconds)",
        color="Algorithm",
        title="Runtime Performance Comparison: Apriori vs FP-Growth",
        text="Time (Seconds)",
        color_discrete_map={
            "Apriori (Level-wise Candidate Gen)": THEME_COLORS["accent_coral"],
            "FP-Growth (FP-Tree Pattern Growth)": THEME_COLORS["accent_green"]
        }
    )
    apply_theme(fig, showlegend=False)
    fig.update_traces(texttemplate="%{y:.4f}s", textposition="outside")
    return fig


def plot_generalization_scatter(gen_df: pd.DataFrame) -> go.Figure:
    """
    Renders Train Confidence vs Test Confidence scatter plot with parity reference line.
    Points on or above the diagonal indicate strong rule generalization without overfitting.
    """
    if gen_df.empty or "train_confidence" not in gen_df.columns:
        fig = go.Figure()
        return apply_theme(fig, title="Rule Generalization: Train vs Test Confidence (No Data)")

    fig = px.scatter(
        gen_df,
        x="train_confidence",
        y="test_confidence",
        color="generalization_status",
        size="train_support",
        hover_name="rule_repr",
        hover_data={
            "train_confidence": ":.3f",
            "test_confidence": ":.3f",
            "confidence_drift": ":.3f",
            "test_lift": ":.2f",
            "generalization_status": True
        },
        title="Rule Generalization: Train vs Held-Out Test Confidence",
        labels={
            "train_confidence": "Training Set Confidence",
            "test_confidence": "Held-Out Test Set Confidence",
            "generalization_status": "Status"
        },
        color_discrete_map={
            "Strong Generalizer": THEME_COLORS["accent_green"],
            "Moderate Generalizer": THEME_COLORS["accent_cyan"],
            "Overfitted": THEME_COLORS["accent_coral"],
            "Sub-Independence (Degraded)": THEME_COLORS["accent_yellow"],
            "Zero Support in Test": THEME_COLORS["text_muted"]
        }
    )

    # Add 45-degree parity line (y = x)
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="#94A3B8", dash="dash", width=1.5),
            name="Ideal Parity (y=x)",
            showlegend=True
        )
    )

    apply_theme(fig)
    fig.update_xaxes(range=[0, 1.05])
    fig.update_yaxes(range=[0, 1.05])
    return fig


def plot_threshold_sensitivity_curves(sens_df: pd.DataFrame) -> go.Figure:
    """
    Renders multi-metric line chart showing Precision, Recall, and F1-Score
    across varying minimum support and confidence thresholds.
    """
    if sens_df.empty:
        fig = go.Figure()
        return apply_theme(fig, title="Threshold Sensitivity Curves (No Data)")

    fig = go.Figure()

    # Format label for x-axis combining support and confidence
    sens_df = sens_df.copy()
    sens_df["threshold_label"] = sens_df.apply(
        lambda r: f"Sup:{r['min_support']:.2f}|Conf:{r['min_confidence']:.2f}", axis=1
    )

    fig.add_trace(go.Scatter(
        x=sens_df["threshold_label"],
        y=sens_df["macro_precision"],
        mode="lines+markers",
        name="Macro Precision",
        line=dict(color=THEME_COLORS["accent_green"], width=2.5)
    ))

    fig.add_trace(go.Scatter(
        x=sens_df["threshold_label"],
        y=sens_df["macro_recall"],
        mode="lines+markers",
        name="Macro Recall",
        line=dict(color=THEME_COLORS["accent_cyan"], width=2.5)
    ))

    fig.add_trace(go.Scatter(
        x=sens_df["threshold_label"],
        y=sens_df["macro_f1"],
        mode="lines+markers",
        name="Macro F1-Score",
        line=dict(color=THEME_COLORS["accent_yellow"], width=3)
    ))

    fig.add_trace(go.Scatter(
        x=sens_df["threshold_label"],
        y=sens_df["test_coverage_pct"] / 100.0,
        mode="lines+markers",
        name="Coverage Proportion",
        line=dict(color=THEME_COLORS["accent_coral"], width=2, dash="dot")
    ))

    apply_theme(
        fig,
        title="Rule Quality Trade-Off Curves Across Hyperparameter Thresholds",
        xaxis_title="Threshold Configuration (Min Support | Min Confidence)",
        yaxis_title="Metric Value (0.0 - 1.0)"
    )
    fig.update_xaxes(tickangle=-45)
    fig.update_yaxes(range=[0, 1.05])
    return fig


def plot_rule_drift_histogram(gen_df: pd.DataFrame) -> go.Figure:
    """
    Renders histogram of rule confidence drift (Test Confidence - Train Confidence).
    """
    if gen_df.empty or "confidence_drift" not in gen_df.columns:
        fig = go.Figure()
        return apply_theme(fig, title="Rule Confidence Drift Distribution (No Data)")

    fig = px.histogram(
        gen_df,
        x="confidence_drift",
        nbins=20,
        title="Distribution of Generalization Confidence Drift (Δ Confidence)",
        labels={"confidence_drift": "Confidence Drift (Test Conf - Train Conf)"},
        color_discrete_sequence=[THEME_COLORS["accent_blue"]]
    )

    # Vertical zero-drift line
    fig.add_vline(x=0.0, line_dash="dash", line_color=THEME_COLORS["accent_green"], annotation_text="Zero Drift (Perfect Stability)")

    apply_theme(fig)
    return fig


def plot_candidate_levels_bar(levels_df: pd.DataFrame) -> go.Figure:
    """
    Renders grouped bar chart showing Apriori level-by-level candidate generation,
    pruning by downward closure, and discovered frequent itemsets.
    """
    if levels_df.empty or "level_name" not in levels_df.columns:
        fig = go.Figure()
        return apply_theme(fig, title="Apriori Level Candidate Pruning (No Data)")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=levels_df["level_name"],
        y=levels_df["theoretical_combinatorial"],
        name="Combinatorial Space (Without Apriori)",
        marker_color="#334155"
    ))

    fig.add_trace(go.Bar(
        x=levels_df["level_name"],
        y=levels_df["scanned_against_database"],
        name="Candidates Scanned (Ck)",
        marker_color=THEME_COLORS["accent_yellow"]
    ))

    fig.add_trace(go.Bar(
        x=levels_df["level_name"],
        y=levels_df["frequent_itemsets_found"],
        name="Frequent Itemsets Found (Lk)",
        marker_color=THEME_COLORS["accent_green"]
    ))

    apply_theme(
        fig,
        title="Apriori Candidate Generation & Downward-Closure Pruning per Level",
        barmode="group",
        yaxis_type="log",
        yaxis_title="Count (Log Scale)"
    )
    return fig


def plot_fptree_compression(stats_dict: Dict[str, Any]) -> go.Figure:
    """
    Renders horizontal comparison illustrating FP-Tree memory compression vs raw transaction items.
    """
    raw_tokens = stats_dict.get("total_raw_item_tokens", 0)
    tree_nodes = stats_dict.get("fp_tree_total_nodes", 0)
    
    comp_df = pd.DataFrame([
        {"Data Structure": "Raw Transaction Item Tokens", "Elements": raw_tokens},
        {"Data Structure": "FP-Tree Compressed Nodes", "Elements": tree_nodes}
    ])

    fig = px.bar(
        comp_df,
        x="Elements",
        y="Data Structure",
        orientation="h",
        title=f"FP-Tree Prefix Sharing Compression: {stats_dict.get('compression_ratio_pct', 0)}% Savings",
        color="Data Structure",
        color_discrete_map={
            "Raw Transaction Item Tokens": THEME_COLORS["accent_coral"],
            "FP-Tree Compressed Nodes": THEME_COLORS["accent_green"]
        },
        text="Elements"
    )
    apply_theme(fig, showlegend=False)
    fig.update_traces(texttemplate="%{x:,} elements", textposition="outside")
    return fig
