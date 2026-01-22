import json
from typing import List, Tuple, Dict


def cluster_addresses(addresses: List[str]) -> Tuple[Dict[str, List[str]], Dict]:
    """Very small clustering by prefix heuristics.

    Returns (clusters, analysis) where `clusters` maps cluster keys to address lists.
    """
    clusters: Dict[str, List[str]] = {}
    for a in addresses:
        key = a[:4] if isinstance(a, str) and len(a) >= 4 else "misc"
        clusters.setdefault(key, []).append(a)

    analysis = {
        "num_addresses": len(addresses),
        "num_clusters": len(clusters)
    }

    out_path = "address_cluster_analysis.json"
    try:
        with open(out_path, "w") as f:
            json.dump({"clusters": clusters, "analysis": analysis}, f, indent=2)
    except Exception:
        pass

    return clusters, analysis


if __name__ == "__main__":
    import sys
    addrs = sys.argv[1:]
    clusters, analysis = cluster_addresses(addrs)
    print(json.dumps({"clusters": clusters, "analysis": analysis}, indent=2))
