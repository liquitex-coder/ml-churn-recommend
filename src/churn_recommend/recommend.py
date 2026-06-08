"""Cross-sell recommender (FR-6, FR-7, FR-22).

Combines two signals:

* item-item collaborative filtering — cosine similarity between products in
  the customer x product purchase matrix (FR-6).
* a NetworkX co-purchase graph — products are nodes, co-purchase counts are
  weighted edges, and graph centrality / weighted neighbour strength provide a
  complementary popularity-aware signal (FR-6).

Recommendations exclude products the customer already owns (FR-22) and the
recommender is intended for active / low-risk customers (FR-7).
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from . import config


def build_customer_product_matrix(purchases: pd.DataFrame) -> pd.DataFrame:
    """Customer x product binary ownership matrix (FR-6 input)."""
    matrix = (
        purchases.assign(owned=1)
        .pivot_table(index="customer_id", columns="product", values="owned", fill_value=0)
    )
    # Ensure every catalog product is a column for stable item-item space.
    for product in config.PRODUCTS:
        if product not in matrix.columns:
            matrix[product] = 0
    return matrix[config.PRODUCTS]


def item_item_similarity(matrix: pd.DataFrame) -> pd.DataFrame:
    """Item-item cosine similarity over products (collaborative filtering, FR-6)."""
    sim = cosine_similarity(matrix.T.to_numpy())
    return pd.DataFrame(sim, index=matrix.columns, columns=matrix.columns)


def build_copurchase_graph(purchases: pd.DataFrame) -> nx.Graph:
    """NetworkX co-purchase graph: products as nodes, co-purchase counts as edges (FR-6)."""
    graph = nx.Graph()
    graph.add_nodes_from(config.PRODUCTS)
    for _, group in purchases.groupby("customer_id"):
        items = sorted(set(group["product"]))
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                if graph.has_edge(a, b):
                    graph[a][b]["weight"] += 1
                else:
                    graph.add_edge(a, b, weight=1)
    return graph


def graph_affinity(graph: nx.Graph, owned: set[str], candidate: str) -> float:
    """Weighted co-purchase affinity of a candidate to the owned set (FR-6)."""
    if candidate not in graph:
        return 0.0
    score = 0.0
    for owned_product in owned:
        if graph.has_edge(owned_product, candidate):
            score += graph[owned_product][candidate]["weight"]
    return float(score)


class CrossSellRecommender:
    """Item-item CF + NetworkX graph cross-sell recommender (FR-6, FR-7, FR-22)."""

    def __init__(self, purchases: pd.DataFrame):
        self.purchases = purchases
        self.matrix = build_customer_product_matrix(purchases)
        self.item_sim = item_item_similarity(self.matrix)
        self.graph = build_copurchase_graph(purchases)
        # Degree centrality gives a popularity prior over products.
        self.centrality = nx.degree_centrality(self.graph)

    def owned_products(self, customer_id: str) -> set[str]:
        """Products the customer already purchased (used for FR-22 exclusion)."""
        if customer_id not in self.matrix.index:
            return set()
        row = self.matrix.loc[customer_id]
        return set(row[row > 0].index)

    def recommend(self, customer_id: str, top_k: int = config.TOP_K_RECOMMENDATIONS) -> pd.DataFrame:
        """Top-K cross-sell recommendations for a customer (FR-6, FR-22).

        Score = item-item CF similarity to owned products + normalised graph
        co-purchase affinity + a small centrality prior. Already-owned products
        are excluded (FR-22).
        """
        owned = self.owned_products(customer_id)
        candidates = [p for p in config.PRODUCTS if p not in owned]  # FR-22 exclusion

        rows = []
        for cand in candidates:
            cf_score = float(self.item_sim.loc[cand, list(owned)].sum()) if owned else 0.0
            g_score = graph_affinity(self.graph, owned, cand)
            cent = self.centrality.get(cand, 0.0)
            rows.append(
                {
                    "product": cand,
                    "cf_score": cf_score,
                    "graph_score": g_score,
                    "centrality": cent,
                }
            )

        recos = pd.DataFrame(rows)
        if recos.empty:
            return recos

        # Normalise graph score so it is comparable to the cosine CF score.
        max_g = recos["graph_score"].max()
        recos["graph_norm"] = recos["graph_score"] / max_g if max_g > 0 else 0.0
        recos["score"] = recos["cf_score"] + recos["graph_norm"] + 0.25 * recos["centrality"]
        recos = recos.sort_values("score", ascending=False).reset_index(drop=True)
        return recos.head(top_k)

    def recommend_for_active_customers(
        self,
        risk_percent: pd.Series,
        top_k: int = config.TOP_K_RECOMMENDATIONS,
        risk_threshold: float = config.HIGH_RISK_THRESHOLD,
    ) -> dict[str, pd.DataFrame]:
        """Generate recommendations targeting active / low-risk customers (FR-7).

        ``risk_percent`` is indexed by customer_id (0-100%). Customers at or
        below ``risk_threshold`` are treated as active / cared-for targets.
        """
        active = risk_percent[risk_percent <= risk_threshold].index
        return {cid: self.recommend(cid, top_k=top_k) for cid in active}
