import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import networkx as nx
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Optional, Tuple, List
import warnings

warnings.filterwarnings("ignore")


class AssociationAnalyzer:
    def __init__(self):
        self.dark_colors = [
            "#FF6B6B",
            "#4ECDC4",
            "#45B7D1",
            "#96CEB4",
            "#FECA57",
            "#FF9FF3",
            "#54A0FF",
            "#5F27CD",
            "#00D2D3",
            "#FF9F43",
            "#A55EEA",
            "#26DE81",
            "#FD79A8",
            "#FDCB6E",
            "#6C5CE7",
            "#A29BFE",
            "#74B9FF",
            "#00B894",
            "#E17055",
            "#81ECEC",
        ]

    def association_graph(
        self,
        df: pd.DataFrame,
        ingredient_col: str = "ingredient_name",
        invoice_col: str = "invoice",
        min_product_count: int = 50,
        top_n_products: int = 160,
        cooccurrence_percentile: float = 75.0,
        lift_threshold: float = 1.1,
        max_connections: int = 150,
        clustering_method: str = "dbscan",  # "dbscan" or "kmeans"
        dbscan_eps_range: Tuple[float, float] = (0.3, 1.0),
        dbscan_min_samples: int = 2,
        kmeans_k_range: Tuple[int, int] = (3, 12),
        max_clusters_display: int = 12,
        figure_width: int = 1500,
        figure_height: int = 1000,
        node_size_multiplier: float = 1.0,
        show_legend: bool = True,
        return_html: bool = True,
    ) -> Optional[str]:
        print(f"Starting analysis with {len(df)} rows...")
        df_clean = df[df[ingredient_col] != "nan"].copy()
        print(f"After removing 'nan': {len(df_clean)} rows")

        product_counts = df_clean[ingredient_col].value_counts()
        popular_products = product_counts[product_counts >= min_product_count].index
        df_filtered = df_clean[df_clean[ingredient_col].isin(popular_products)]

        top_products = (
            df_filtered[ingredient_col].value_counts().nlargest(top_n_products).index
        )
        df_final = df_filtered[df_filtered[ingredient_col].isin(top_products)]

        print(f"Working with {len(top_products)} products after filtering")

        if len(top_products) < 2:
            print("Error: Not enough products after filtering!")
            return None

        transactions = (
            df_final.groupby(invoice_col)[ingredient_col].apply(list).tolist()
        )
        transactions = [t for t in transactions if len(t) > 1]

        print(f"Number of multi-item transactions: {len(transactions)}")

        if len(transactions) < 10:
            print("Error: Not enough multi-item transactions!")
            return None

        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        basket_df = pd.DataFrame(te_ary, columns=te.columns_)

        print(f"Basket matrix shape: {basket_df.shape}")

        cooccurrence = basket_df.T.dot(basket_df).astype(int)
        np.fill_diagonal(cooccurrence.values, 0)

        support = basket_df.sum() / len(basket_df)

        lift_matrix = pd.DataFrame(
            index=basket_df.columns, columns=basket_df.columns, dtype=float
        )

        for i, prod1 in enumerate(basket_df.columns):
            for j, prod2 in enumerate(basket_df.columns):
                if i != j:
                    joint_prob = cooccurrence.iloc[i, j] / len(basket_df)
                    expected = support.iloc[i] * support.iloc[j]
                    lift_matrix.iloc[i, j] = (
                        joint_prob / expected if expected > 0 else 0
                    )
                else:
                    lift_matrix.iloc[i, j] = 0

        nonzero_cooc = cooccurrence.values[cooccurrence.values > 0]
        cooc_threshold = (
            np.percentile(nonzero_cooc, cooccurrence_percentile)
            if len(nonzero_cooc) > 0
            else 1
        )

        print(
            f"Co-occurrence threshold: {cooc_threshold}, Lift threshold: {lift_threshold}"
        )

        G = nx.Graph()

        for product in basket_df.columns:
            freq = int(support[product] * len(basket_df))
            G.add_node(product, frequency=freq, support=support[product])

        edge_candidates = []
        for i, prod1 in enumerate(basket_df.columns):
            for j, prod2 in enumerate(basket_df.columns):
                if i < j:
                    cooc_count = cooccurrence.iloc[i, j]
                    lift_val = lift_matrix.iloc[i, j]
                    if cooc_count >= cooc_threshold and lift_val >= lift_threshold:
                        G.add_edge(
                            prod1, prod2, weight=lift_val, cooccurrence=int(cooc_count)
                        )
                    elif cooc_count > 0:
                        edge_candidates.append(
                            (prod1, prod2, cooc_count, lift_val, cooc_count * lift_val)
                        )

        print(
            f"Initial network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
        )

        if G.number_of_edges() < 50 and edge_candidates:
            print("Network sparse, adding top connections...")
            edge_candidates.sort(key=lambda x: x[4], reverse=True)
            edges_to_add = min(max_connections, len(edge_candidates))

            for prod1, prod2, cooc_count, lift_val, _ in edge_candidates[:edges_to_add]:
                if not G.has_edge(prod1, prod2):
                    G.add_edge(
                        prod1, prod2, weight=lift_val, cooccurrence=int(cooc_count)
                    )

        isolated = list(nx.isolates(G))
        G.remove_nodes_from(isolated)
        print(f"Removed {len(isolated)} isolated nodes")

        if G.number_of_nodes() < 3:
            print("Error: Network too small for clustering!")
            return None

        node_list = list(G.nodes())
        features = []

        try:
            betweenness = nx.betweenness_centrality(G)
            closeness = nx.closeness_centrality(G)
            clustering_coef = nx.clustering(G)
        except:
            betweenness = {node: 0 for node in G.nodes()}
            closeness = {node: 0 for node in G.nodes()}
            clustering_coef = {node: 0 for node in G.nodes()}

        for node in node_list:
            degree = G.degree(node)
            frequency = G.nodes[node].get("frequency", 1)
            support_val = G.nodes[node].get("support", 0.001)

            neighbors = list(G.neighbors(node))
            avg_lift = (
                np.mean([G[node][neighbor].get("weight", 1) for neighbor in neighbors])
                if neighbors
                else 0
            )

            features.append(
                [
                    degree,
                    betweenness.get(node, 0),
                    closeness.get(node, 0),
                    clustering_coef.get(node, 0),
                    frequency,
                    support_val,
                    avg_lift,
                    len(neighbors),
                ]
            )

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        best_labels = None
        best_n_clusters = 0
        method_used = ""

        if clustering_method.lower() == "dbscan":
            print("Trying DBSCAN clustering...")
            eps_values = np.linspace(dbscan_eps_range[0], dbscan_eps_range[1], 8)

            for eps in eps_values:
                dbscan = DBSCAN(eps=eps, min_samples=dbscan_min_samples)
                labels = dbscan.fit_predict(features_scaled)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)

                if 3 <= n_clusters <= 10 and n_noise < len(labels) * 0.3:
                    if n_clusters > best_n_clusters:
                        best_labels = labels
                        best_n_clusters = n_clusters
                        method_used = f"DBSCAN (eps={eps:.2f})"

            if best_labels is None:
                print("DBSCAN failed, falling back to K-means...")
                clustering_method = "kmeans"

        if clustering_method.lower() == "kmeans" or best_labels is None:
            print("Using K-means clustering...")
            k_min, k_max = kmeans_k_range
            k_max = min(k_max, len(node_list) // 2)

            if k_max <= k_min:
                k_max = k_min + 2

            inertias = []
            k_range = range(k_min, k_max + 1)

            for k in k_range:
                if k <= len(node_list):
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    kmeans.fit(features_scaled)
                    inertias.append(kmeans.inertia_)
                else:
                    break

            if len(inertias) > 1:
                diffs = [
                    inertias[i] - inertias[i + 1] for i in range(len(inertias) - 1)
                ]
                best_k = k_range[diffs.index(max(diffs))] if diffs else k_min
            else:
                best_k = k_min

            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            best_labels = kmeans.fit_predict(features_scaled)
            best_n_clusters = best_k
            method_used = f"K-means (k={best_k})"

        node_to_cluster = {}
        noise_cluster_id = best_n_clusters

        for i, node in enumerate(node_list):
            cluster_id = best_labels[i]
            if cluster_id == -1:
                cluster_id = noise_cluster_id
                noise_cluster_id += 1
            node_to_cluster[node] = cluster_id

        print(f"Final clustering: {best_n_clusters} main clusters using {method_used}")

        if G.number_of_nodes() <= 40:
            pos = nx.spring_layout(G, seed=42, k=4, iterations=200)
        else:
            pos = nx.fruchterman_reingold_layout(G, seed=42, iterations=100)

        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.8, color="rgba(200,200,200,0.4)"),
            hoverinfo="none",
            mode="lines",
        )

        node_x, node_y, node_text, node_color, node_info, node_sizes = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        frequencies = [G.nodes[node].get("frequency", 1) for node in G.nodes()]
        max_freq = max(frequencies) if frequencies else 1

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            display_name = node[:12] + "..." if len(node) > 12 else node
            node_text.append(display_name)

            cluster_id = node_to_cluster.get(node, 0)
            node_color.append(self.dark_colors[cluster_id % len(self.dark_colors)])

            freq = G.nodes[node].get("frequency", 1)
            degree = G.degree(node)
            size = (10 + (freq / max_freq) * 15 + degree * 2) * node_size_multiplier
            node_sizes.append(min(size, 50))

            neighbors = list(G.neighbors(node))[:5]
            node_info.append(
                f"<b>{node}</b><br>"
                f"Cluster: {cluster_id}<br>"
                f"Frequency: {freq}<br>"
                f"Connections: {degree}<br>"
                f"Connected to: {', '.join(neighbors)}"
            )

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="middle center",
            textfont=dict(size=8, color="white"),
            marker=dict(
                color=node_color,
                size=node_sizes,
                line=dict(width=2, color="rgba(50,50,50,0.8)"),
            ),
            hoverinfo="text",
            hovertext=node_info,
        )

        legend_traces = []
        if show_legend:
            cluster_stats = {}
            for node, cluster_id in node_to_cluster.items():
                if cluster_id not in cluster_stats:
                    cluster_stats[cluster_id] = []
                cluster_stats[cluster_id].append(node)

            main_clusters = {k: v for k, v in cluster_stats.items() if len(v) >= 2}

            for cluster_id in sorted(main_clusters.keys())[:max_clusters_display]:
                cluster_size = len(main_clusters[cluster_id])
                legend_traces.append(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="markers",
                        marker=dict(
                            size=12,
                            color=self.dark_colors[cluster_id % len(self.dark_colors)],
                        ),
                        showlegend=True,
                        name=f"Cluster {cluster_id} ({cluster_size} items)",
                    )
                )

        fig = go.Figure(
            data=[edge_trace, node_trace] + legend_traces,
            layout=go.Layout(
                title=dict(
                    text=f"Association Network ({method_used})<br>"
                    f"<sub>{len(G.nodes())} items, {len(G.edges())} associations, {best_n_clusters} clusters</sub>",
                    x=0.5,
                    font=dict(color="white", size=16),
                ),
                showlegend=show_legend,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=80),
                paper_bgcolor="rgba(30,30,30,1)",
                plot_bgcolor="rgba(30,30,30,1)",
                xaxis=dict(
                    showgrid=False, zeroline=False, showticklabels=False, color="white"
                ),
                yaxis=dict(
                    showgrid=False, zeroline=False, showticklabels=False, color="white"
                ),
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.01,
                    bgcolor="rgba(50,50,50,0.8)",
                    bordercolor="rgba(200,200,200,0.3)",
                    borderwidth=1,
                    font=dict(color="white"),
                ),
                width=figure_width,
                height=figure_height,
                annotations=[
                    dict(
                        text=f"Method: {method_used} | Lift threshold: {lift_threshold} | Min frequency: {min_product_count}",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.02,
                        y=0.02,
                        font=dict(size=10, color="rgba(200,200,200,0.7)"),
                    )
                ],
            ),
        )

        if return_html:
            return fig.to_html(include_plotlyjs=True, div_id="association-network")
        else:
            fig.show()
            return None

    def association_rules(
        self,
        df: pd.DataFrame,
        brand_col: str = "brand_name",
        invoice_col: str = "invoice",
        min_support: float = 0.01,
        min_confidence: float = 0.1,
        min_lift: float = 1.0,
        max_len: int = 2,
        top_n_brands: int = 200,
        min_brand_count: int = 20,
        sort_by: str = "lift",
        max_rules_display: int = 100,
        figure_width: int = 1400,
        figure_height: int = 800,
        return_html: bool = True,
        include_metrics_plot: bool = True,
    ) -> Optional[str]:
        from mlxtend.frequent_patterns import apriori, association_rules as mlx_rules

        print(f"Starting association rules analysis with {len(df)} rows...")

        df_clean = df[df[brand_col].notna() & (df[brand_col] != "nan")].copy()
        print(f"After removing null/nan brands: {len(df_clean)} rows")

        if len(df_clean) == 0:
            print("Error: No valid brand data found!")
            return None

        brand_counts = df_clean[brand_col].value_counts()
        popular_brands = brand_counts[brand_counts >= min_brand_count].index
        df_filtered = df_clean[df_clean[brand_col].isin(popular_brands)]

        top_brands = df_filtered[brand_col].value_counts().nlargest(top_n_brands).index
        df_final = df_filtered[df_filtered[brand_col].isin(top_brands)]

        print(f"Working with {len(top_brands)} brands after filtering")
        print(f"Final dataset: {len(df_final)} rows")

        if len(top_brands) < 2:
            print("Error: Not enough brands after filtering!")
            return None

        transactions = df_final.groupby(invoice_col)[brand_col].apply(list).tolist()
        transactions = [list(set(t)) for t in transactions]
        transactions = [t for t in transactions if len(t) >= 1]

        print(f"Total transactions: {len(transactions)}")
        print(
            f"Multi-item transactions: {len([t for t in transactions if len(t) > 1])}"
        )

        if len(transactions) < 10:
            print("Error: Not enough transactions!")
            return None

        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        basket_df = pd.DataFrame(te_ary, columns=te.columns_)

        print(f"Basket matrix shape: {basket_df.shape}")
        print(f"Average basket size: {basket_df.sum(axis=1).mean():.2f}")

        print(f"Finding frequent itemsets with min_support={min_support}...")
        frequent_itemsets = apriori(
            basket_df, min_support=min_support, use_colnames=True, max_len=max_len
        )

        if len(frequent_itemsets) == 0:
            print(f"No frequent itemsets found with min_support={min_support}")
            print("Try lowering the min_support parameter")
            return None

        print(f"Found {len(frequent_itemsets)} frequent itemsets")

        print(
            f"Generating rules with min_confidence={min_confidence}, min_lift={min_lift}..."
        )

        try:
            rules = mlx_rules(
                frequent_itemsets,
                metric="confidence",
                min_threshold=min_confidence,
                num_itemsets=len(frequent_itemsets),
            )

            rules = rules[rules["lift"] >= min_lift]

            if len(rules) == 0:
                print("No rules found with current thresholds")
                print("Try lowering min_confidence or min_lift")
                return None

        except Exception as e:
            print(f"Error generating rules: {e}")
            return None

        rules["conviction"] = (1 - rules["consequent support"]) / (
            1 - rules["confidence"]
        )
        rules["conviction"] = (
            rules["conviction"].replace([np.inf, -np.inf], np.nan).fillna(999)
        )

        rules["antecedents_str"] = rules["antecedents"].apply(
            lambda x: ", ".join(list(x))
        )
        rules["consequents_str"] = rules["consequents"].apply(
            lambda x: ", ".join(list(x))
        )
        rules["rule"] = rules["antecedents_str"] + " → " + rules["consequents_str"]

        if sort_by in rules.columns:
            rules = rules.sort_values(sort_by, ascending=False)
        else:
            rules = rules.sort_values("lift", ascending=False)

        print(f"Generated {len(rules)} association rules")

        display_rules = rules.head(max_rules_display).copy()

        if include_metrics_plot and len(display_rules) > 1:
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    "Support vs Confidence",
                    "Lift vs Confidence",
                    "Top Rules by Lift",
                    "Support vs Lift",
                ),
                specs=[
                    [{"secondary_y": False}, {"secondary_y": False}],
                    [{"colspan": 2}, None],
                ],
                vertical_spacing=0.12,
                horizontal_spacing=0.1,
            )

            colors = px.colors.sequential.Viridis

            fig.add_trace(
                go.Scatter(
                    x=display_rules["support"],
                    y=display_rules["confidence"],
                    mode="markers",
                    marker=dict(
                        size=display_rules["lift"] * 5,
                        color=display_rules["lift"],
                        colorscale="Viridis",
                        showscale=False,
                        line=dict(width=1, color="white"),
                    ),
                    text=display_rules["rule"],
                    hovertemplate="<b>%{text}</b><br>"
                    + "Support: %{x:.3f}<br>"
                    + "Confidence: %{y:.3f}<br>"
                    + "Lift: %{marker.color:.2f}<extra></extra>",
                    name="Rules",
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=display_rules["lift"],
                    y=display_rules["confidence"],
                    mode="markers",
                    marker=dict(
                        size=display_rules["support"] * 200,
                        color=display_rules["support"],
                        colorscale="Plasma",
                        showscale=False,
                        line=dict(width=1, color="white"),
                    ),
                    text=display_rules["rule"],
                    hovertemplate="<b>%{text}</b><br>"
                    + "Lift: %{x:.2f}<br>"
                    + "Confidence: %{y:.3f}<br>"
                    + "Support: %{marker.color:.3f}<extra></extra>",
                    name="Rules",
                ),
                row=1,
                col=2,
            )

            top_rules = display_rules.head(20)
            fig.add_trace(
                go.Bar(
                    y=top_rules["rule"],
                    x=top_rules["lift"],
                    orientation="h",
                    marker=dict(
                        color=top_rules["lift"],
                        colorscale="Turbo",
                        line=dict(width=1, color="white"),
                    ),
                    text=[f"{lift:.2f}" for lift in top_rules["lift"]],
                    textposition="inside",
                    hovertemplate="<b>%{y}</b><br>"
                    + "Lift: %{x:.2f}<br>"
                    + "Confidence: %{customdata[0]:.3f}<br>"
                    + "Support: %{customdata[1]:.3f}<extra></extra>",
                    customdata=top_rules[["confidence", "support"]].values,
                    name="Top Rules",
                ),
                row=2,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=display_rules["support"],
                    y=display_rules["lift"],
                    mode="markers",
                    marker=dict(
                        size=10,
                        color=display_rules["confidence"],
                        colorscale="Cividis",
                        showscale=False,
                        line=dict(width=1, color="white"),
                    ),
                    text=display_rules["rule"],
                    hovertemplate="<b>%{text}</b><br>"
                    + "Support: %{x:.3f}<br>"
                    + "Lift: %{y:.2f}<br>"
                    + "Confidence: %{marker.color:.3f}<extra></extra>",
                    name="Rules",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

            fig.update_layout(
                title=dict(
                    text=f"Association Rules Analysis - {len(rules)} Rules Found<br>"
                    + f"<sub>Min Support: {min_support}, Min Confidence: {min_confidence}, Min Lift: {min_lift}</sub>",
                    x=0.5,
                    font=dict(color="white", size=16),
                ),
                paper_bgcolor="rgba(30,30,30,1)",
                plot_bgcolor="rgba(30,30,30,1)",
                font=dict(color="white"),
                showlegend=False,
                width=figure_width,
                height=figure_height,
                margin=dict(l=50, r=50, t=100, b=50),
            )

            fig.update_xaxes(
                showgrid=True,
                gridcolor="rgba(100,100,100,0.3)",
                color="white",
                title_font=dict(color="white"),
            )
            fig.update_yaxes(
                showgrid=True,
                gridcolor="rgba(100,100,100,0.3)",
                color="white",
                title_font=dict(color="white"),
            )

            for i in fig["layout"]["annotations"]:
                i["font"] = dict(color="white", size=12)

            fig.update_xaxes(title_text="Support", row=1, col=1)
            fig.update_yaxes(title_text="Confidence", row=1, col=1)
            fig.update_xaxes(title_text="Lift", row=1, col=2)
            fig.update_yaxes(title_text="Confidence", row=1, col=2)
            fig.update_xaxes(title_text="Lift", row=2, col=1)
            fig.update_yaxes(title_text="Rules (Top 20)", row=2, col=1)

        else:
            fig = go.Figure(
                data=[
                    go.Table(
                        header=dict(
                            values=[
                                "Rule",
                                "Support",
                                "Confidence",
                                "Lift",
                                "Conviction",
                            ],
                            fill_color="rgba(50,50,50,0.8)",
                            align="left",
                            font=dict(color="white", size=12),
                        ),
                        cells=dict(
                            values=[
                                display_rules["rule"],
                                [f"{x:.4f}" for x in display_rules["support"]],
                                [f"{x:.4f}" for x in display_rules["confidence"]],
                                [f"{x:.2f}" for x in display_rules["lift"]],
                                [
                                    f"{x:.2f}" if x != 999 else "∞"
                                    for x in display_rules["conviction"]
                                ],
                            ],
                            fill_color="rgba(30,30,30,0.8)",
                            align="left",
                            font=dict(color="white", size=10),
                        ),
                    )
                ]
            )

            fig.update_layout(
                title=f"Association Rules - Top {len(display_rules)} Rules",
                paper_bgcolor="rgba(30,30,30,1)",
                font=dict(color="white"),
                width=figure_width,
                height=figure_height,
            )

        print("\n=== ASSOCIATION RULES SUMMARY ===")
        print(f"Total rules generated: {len(rules)}")
        print(f"Average support: {rules['support'].mean():.4f}")
        print(f"Average confidence: {rules['confidence'].mean():.4f}")
        print(f"Average lift: {rules['lift'].mean():.2f}")
        print(f"Rules sorted by: {sort_by}")

        print(f"\n=== TOP 10 RULES BY {sort_by.upper()} ===")
        for idx, row in display_rules.head(10).iterrows():
            print(f"{row['rule']}")
            print(
                f"  Support: {row['support']:.4f} | Confidence: {row['confidence']:.4f} | Lift: {row['lift']:.2f}"
            )
            print()

        if return_html:
            html_content = fig.to_html(
                include_plotlyjs=True, div_id="association-rules"
            )

            summary_html = f"""
            <div style="background-color: #1e1e1e; color: white; padding: 20px; margin-top: 20px; border-radius: 8px;">
                <h3>Association Rules Summary</h3>
                <p><strong>Total Rules:</strong> {len(rules)} | <strong>Avg Support:</strong> {rules["support"].mean():.4f} | 
                   <strong>Avg Confidence:</strong> {rules["confidence"].mean():.4f} | <strong>Avg Lift:</strong> {rules["lift"].mean():.2f}</p>
                
                <h4>Top 5 Rules by {sort_by.title()}:</h4>
                <table style="color: white; border-collapse: collapse; width: 100%;">
                    <tr style="border-bottom: 1px solid #444;">
                        <th style="text-align: left; padding: 8px;">Rule</th>
                        <th style="text-align: right; padding: 8px;">Support</th>
                        <th style="text-align: right; padding: 8px;">Confidence</th>
                        <th style="text-align: right; padding: 8px;">Lift</th>
                    </tr>
            """

            for idx, row in display_rules.head(5).iterrows():
                summary_html += f"""
                    <tr style="border-bottom: 1px solid #333;">
                        <td style="padding: 8px;">{row["rule"]}</td>
                        <td style="text-align: right; padding: 8px;">{row["support"]:.4f}</td>
                        <td style="text-align: right; padding: 8px;">{row["confidence"]:.4f}</td>
                        <td style="text-align: right; padding: 8px;">{row["lift"]:.2f}</td>
                    </tr>
                """

            summary_html += """
                </table>
            </div>
            """

            return html_content + summary_html
        else:
            fig.show()
            return None


def main():
    import pandas as pd

    print("Loading data...")
    df = pd.read_parquet("./assets/supertable.parquet")

    analyzer = AssociationAnalyzer()

    # Test Association Network
    print("\n=== Testing Association Network ===")
    html_network = analyzer.association_graph(
        df=df,
        ingredient_col="ingredient_name",
        invoice_col="invoice",
        min_product_count=30,
        top_n_products=200,
        cooccurrence_percentile=70.0,
        lift_threshold=1.2,
        clustering_method="dbscan",
        figure_width=1200,
        figure_height=900,
        return_html=True,
    )
    if html_network:
        with open("test_association_network.html", "w") as f:
            f.write(html_network)
        print("Association network HTML saved to 'test_association_network.html'")
    else:
        print("Failed to generate association network.")

    # Test Association Rules
    print("\n=== Testing Association Rules ===")
    html_rules = analyzer.association_rules(
        df=df,
        brand_col="brand_name",
        invoice_col="invoice",
        min_support=0.005,
        min_confidence=0.15,
        min_lift=1.1,
        max_len=2,
        top_n_brands=150,
        min_brand_count=25,
        sort_by="lift",
        max_rules_display=50,
        figure_width=1200,
        figure_height=800,
        return_html=True,
        include_metrics_plot=True,
    )
    if html_rules:
        with open("test_association_rules.html", "w") as f:
            f.write(html_rules)
        print("Association rules HTML saved to 'test_association_rules.html'")
    else:
        print("Failed to generate association rules.")


if __name__ == "__main__":
    main()
