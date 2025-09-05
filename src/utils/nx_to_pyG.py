#!/usr/bin/env python3
# Convert a NetworkX administrative graph (from get_graph) to a PyTorch Geometric Data object.
# - Builds contiguous node indexing
# - Extracts node features from attributes (default: lon, lat, area)
# - Computes optional edge weights (default: inverse centroid distance in meters)
# - Adds 'pos' (lon, lat) for plotting
# - Preserves node_id mapping and graph metadata

import argparse
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch_geometric.data import Data
import networkx as nx


def _as_float32(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x, dtype=np.float32))


def _build_node_index(G: nx.Graph) -> Tuple[List[Union[int, str]], dict]:
    node_ids = list(G.nodes())
    idx_map = {nid: i for i, nid in enumerate(node_ids)}
    return node_ids, idx_map


def _collect_node_features(
    G: nx.Graph,
    node_ids: Sequence[Union[int, str]],
    feature_keys: Sequence[str] = ("lon", "lat", "area"),
    require_all: bool = False,
) -> torch.Tensor:
    """
    Extracts features in the order of feature_keys; if a key is missing and require_all=False,
    fills with zeros; if require_all=True, raises a KeyError.
    """
    feats = []
    for nid in node_ids:
        attrs = G.nodes[nid]
        row = []
        for k in feature_keys:
            if k in attrs and attrs[k] is not None:
                row.append(float(attrs[k]))
            else:
                if require_all:
                    raise KeyError(f"Missing node attribute '{k}' on node {nid}")
                row.append(0.0)
        feats.append(row)
    x = np.asarray(feats, dtype=np.float32)
    return _as_float32(x)  # [N, F]


def _build_edge_index(
    G: nx.Graph,
    idx_map: dict,
    make_undirected: bool = True,
    add_self_loops: bool = False,
) -> torch.Tensor:
    edges = []
    for u, v in G.edges():
        iu, iv = idx_map[u], idx_map[v]
        edges.append([iu, iv])
        if make_undirected:
            edges.append([iv, iu])
    if add_self_loops:
        for n in idx_map.values():
            edges.append([n, n])
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # [2, E]
    return edge_index


def _edge_weights(
    G: nx.Graph,
    edge_index: torch.Tensor,
    node_ids: Sequence[Union[int, str]],
    mode: str = "inv_dist",  # {'inv_dist','gauss_dist','const1'}
    sigma: float = 25000.0,  # meters, used for gauss_dist
    eps: float = 1.0,        # meters, min distance to avoid div-by-zero
) -> torch.Tensor:
    """
    Compute edge_weight aligned with edge_index:
      - inv_dist: w = 1 / max(d, eps)
      - gauss_dist: w = exp(-(d/sigma)^2)
      - const1: w = 1 for all edges
    Distances use 'x','y' node attributes (meters; produced by get_graph with EPSG:2056).
    """
    N = len(node_ids)
    xy = np.zeros((N, 2), dtype=np.float64)
    has_xy = True
    for i, nid in enumerate(node_ids):
        d = G.nodes[nid]
        if "x" in d and "y" in d and d["x"] is not None and d["y"] is not None:
            xy[i, 0] = float(d["x"])
            xy[i, 1] = float(d["y"])
        else:
            has_xy = False
            break

    if mode == "const1" or not has_xy or edge_index.numel() == 0:
        return torch.ones(edge_index.size(1), dtype=torch.float32)

    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    diff = xy[src] - xy[dst]
    d = np.sqrt((diff ** 2).sum(axis=1))  # meters

    if mode == "inv_dist":
        w = 1.0 / np.maximum(d, eps)
    elif mode == "gauss_dist":
        # exp(-(d/sigma)^2)
        sigma = float(sigma) if sigma > 0 else 1.0
        w = np.exp(- (d / sigma) ** 2)
    else:
        raise ValueError(f"Unknown edge weight mode: {mode}")

    return _as_float32(w)


def _node_pos_lonlat(G: nx.Graph, node_ids: Sequence[Union[int, str]]) -> torch.Tensor:
    """
    Returns node positions as (lon, lat) in float32 for visualization with PyG.
    Missing values are filled with zeros.
    """
    pos = np.zeros((len(node_ids), 2), dtype=np.float32)
    for i, nid in enumerate(node_ids):
        d = G.nodes[nid]
        lon = float(d["lon"]) if "lon" in d and d["lon"] is not None else 0.0
        lat = float(d["lat"]) if "lat" in d and d["lat"] is not None else 0.0
        pos[i, 0] = lon
        pos[i, 1] = lat
    return _as_float32(pos)


def nx_to_pyg(
    G: nx.Graph,
    node_feature_keys: Sequence[str] = ("lon", "lat", "area"),
    edge_weight_mode: str = "inv_dist",   # {'inv_dist','gauss_dist','const1'}
    gauss_sigma_m: float = 25000.0,
    add_self_loops: bool = False,
    make_undirected: bool = True,
    normalize_x: bool = False,
) -> Data:
    """
    Convert a NetworkX graph (as built by get_graph) to a torch_geometric.data.Data object.

    Returns a Data with:
      - x: [N, F] node feature tensor (float32)
      - edge_index: [2, E] long tensor
      - edge_weight: [E] float32 tensor (optional weighting)
      - pos: [N, 2] (lon, lat) float32 (for visualization)
      - node_ids: list (Python) of original IDs
      - graph metadata copied into attributes (level, rook, id_col, name_col)
    """
    node_ids, idx_map = _build_node_index(G)
    x = _collect_node_features(G, node_ids, node_feature_keys, require_all=False)
    if normalize_x and x.numel() > 0:
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
        x = (x - mean) / std

    edge_index = _build_edge_index(G, idx_map, make_undirected=make_undirected, add_self_loops=add_self_loops)
    edge_weight = _edge_weights(G, edge_index, node_ids, mode=edge_weight_mode, sigma=gauss_sigma_m)
    pos = _node_pos_lonlat(G, node_ids)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
        pos=pos,
    )

    # Preserve useful metadata/mapping
    data.node_ids = node_ids  # list of original node identifiers
    data.level = G.graph.get("level", None)
    data.rook = G.graph.get("rook", None)
    data.id_col = G.graph.get("id_col", None)
    data.name_col = G.graph.get("name_col", None)
    data.crs_nodes = G.graph.get("crs_nodes", None)

    return data


# ------------------------------ CLI (example) ------------------------------

def _demo_args():
    p = argparse.ArgumentParser(description="Convert a NetworkX admin graph to PyG Data")
    p.add_argument("--edge-weight", default="inv_dist", choices=["inv_dist", "gauss_dist", "const1"])
    p.add_argument("--sigma", type=float, default=25000.0, help="Gaussian sigma (meters) for gauss_dist")
    p.add_argument("--normalize-x", action="store_true", help="Z-normalize node features")
    p.add_argument("--save", type=str, default=None, help="Path to save a .pt with the Data object")
    return p.parse_args()


if __name__ == "__main__":
    # Example usage:
    # Here we create a tiny toy graph if G is not provided externally.
    # Replace this with your real import and graph creation.
    G = nx.Graph()
    # toy 3-node line with fake coords in LV95 meters and lon/lat
    G.add_node(100, lon=8.54, lat=47.37, x=2680000.0, y=1240000.0, area=1.0)
    G.add_node(101, lon=8.60, lat=47.40, x=2685000.0, y=1243000.0, area=1.1)
    G.add_node(102, lon=8.66, lat=47.43, x=2690000.0, y=1246000.0, area=0.9)
    G.add_edge(100, 101)
    G.add_edge(101, 102)
    G.graph["level"] = "district"
    G.graph["rook"] = True
    G.graph["id_col"] = "BEZIRKSNUM"
    G.graph["name_col"] = "NAME"
    G.graph["crs_nodes"] = "EPSG:2056"

    args = _demo_args()
    data = nx_to_pyg(
        G,
        node_feature_keys=("lon", "lat", "area"),
        edge_weight_mode=args.edge_weight,
        gauss_sigma_m=args.sigma,
        normalize_x=args.normalize_x,
    )

    print(data)
    if args.save:
        torch.save(data, args.save)
        print(f"Saved PyG Data to: {args.save}")
