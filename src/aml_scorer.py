"""
AML network analysis and risk scoring module for FraudShield.

Builds transaction graphs, detects mule-account chains, and computes
AML-specific risk scores based on network topology.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx


def build_transaction_graph(
    transactions: pd.DataFrame,
    source_col: str = "sender_id",
    target_col: str = "receiver_id",
    weight_col: Optional[str] = "amount",
) -> nx.DiGraph:
    """Construct a directed transaction graph from a ledger of transfers.

    Nodes are accounts (senders and receivers). Edges represent transactions
    and can be weighted by total transfer volume.

    Parameters
    ----------
    transactions : pd.DataFrame
        Ledger of transactions with sender/receiver columns.
    source_col : str
        Column identifying the sending account.
    target_col : str
        Column identifying the receiving account.
    weight_col : str or None
        Column to use as edge weight. If None, edges are unweighted.

    Returns
    -------
    nx.DiGraph
        Directed graph of financial flows.
    """
    G = nx.DiGraph()

    if weight_col and weight_col in transactions.columns:
        # Aggregate total flow between each (sender, receiver) pair
        flow = (
            transactions.groupby([source_col, target_col])[weight_col]
            .sum()
            .reset_index()
        )
        for _, row in flow.iterrows():
            G.add_edge(row[source_col], row[target_col], weight=row[weight_col])
    else:
        edges = zip(transactions[source_col], transactions[target_col])
        G.add_edges_from(edges)

    return G


def detect_mule_chain(
    G: nx.DiGraph,
    max_depth: int = 5,
    min_flow: float = 0.0,
) -> List[List[str]]:
    """Identify potential mule-account chains in the transaction graph.

    Mule chains are directed paths of length >= 2 where accounts pass funds
    onward shortly after receiving them, often indicated by high degree
    centrality and rapid fund turnover.

    Parameters
    ----------
    G : nx.DiGraph
        Directed transaction graph.
    max_depth : int
        Maximum path length (number of hops) to consider.
    min_flow : float
        Minimum edge weight (total flow) to include in search.

    Returns
    -------
    List[List[str]]
        List of mule-chain paths, each an ordered list of account IDs.
    """
    # TODO: implement chain detection (e.g., DFS with pruning)
    chains: List[List[str]] = []
    return chains


def calculate_aml_risk_score(
    G: nx.DiGraph,
    node: str,
) -> Dict[str, float]:
    """Compute an AML risk score for a given account based on network features.

    Combines graph centrality metrics — betweenness, PageRank, clustering
    coefficient, degree — into a composite AML risk score. Accounts that act
    as hubs, bridges, or rapid pass-through nodes receive higher scores.

    Parameters
    ----------
    G : nx.DiGraph
        Directed transaction graph.
    node : str
        Account identifier to score.

    Returns
    -------
    Dict[str, float]
        Dictionary of individual graph metrics and the composite ``aml_score``.
    """
    # TODO: implement composite AML scoring
    score: Dict[str, float] = {
        "in_degree": float(G.in_degree(node)) if node in G else 0.0,
        "out_degree": float(G.out_degree(node)) if node in G else 0.0,
        "aml_score": 0.0,
    }
    return score
