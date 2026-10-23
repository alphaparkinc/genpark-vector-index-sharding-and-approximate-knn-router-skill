"""Example usage for Vector Index Sharding & Approximate KNN Router."""
from client import VectorIndexShardRouter

if __name__ == "__main__":
    centroids = {
        "shard_east": [0.9, 0.1, 0.4],
        "shard_west": [-0.5, 0.8, -0.2],
        "shard_central": [0.85, 0.05, 0.45]
    }
    res = VectorIndexShardRouter.route_query_to_shards([1.0, 0.0, 0.5], centroids, top_k_shards=2)
    print("Selected Shards:", res["selected_shards"])
