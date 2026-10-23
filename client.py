"""Vector Index Sharding & Approximate KNN Router.
100% Python Standard Library.
"""

import math

class VectorIndexShardRouter:
    """Routes high-dimensional embedding search queries across partitioned vector index shards via centroid hashing."""

    @staticmethod
    def route_query_to_shards(query_vector: list, shard_centroids: dict, top_k_shards: int = 2) -> dict:
        similarities = {}
        for shard_id, centroid in shard_centroids.items():
            dot = sum(a * b for a, b in zip(query_vector, centroid))
            norm_q = math.sqrt(sum(a * a for a in query_vector))
            norm_c = math.sqrt(sum(b * b for b in centroid))
            cos_sim = dot / (max(1e-9, norm_q * norm_c))
            similarities[shard_id] = round(cos_sim, 4)

        ranked = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        selected = [s[0] for s in ranked[:top_k_shards]]

        return {
            "total_shards": len(shard_centroids),
            "selected_shards": selected,
            "routing_scores": dict(ranked[:top_k_shards])
        }
