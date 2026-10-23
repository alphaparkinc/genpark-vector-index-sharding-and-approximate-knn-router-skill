"""MCP server for Vector Index Sharding & Approximate KNN Router."""
import sys
import json
from client import VectorIndexShardRouter

def handle_request(req):
    method = req.get("method")
    if method == "tools/list":
        return {
            "tools": [{
                "name": "route_vector_shards",
                "description": "Calculates cosine similarity to shard centroids to route KNN queries",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query_vector": {"type": "array"},
                        "shard_centroids": {"type": "object"},
                        "top_k_shards": {"type": "integer"}
                    },
                    "required": ["query_vector", "shard_centroids"]
                }
            }]
        }
    elif method == "tools/call":
        params = req.get("params", {})
        if params.get("name") == "route_vector_shards":
            args = params.get("arguments", {})
            res = VectorIndexShardRouter.route_query_to_shards(
                args.get("query_vector", []),
                args.get("shard_centroids", {}),
                int(args.get("top_k_shards", 2))
            )
            return {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]}
    return {"error": "Method not found"}

if __name__ == "__main__":
    for line in sys.stdin:
        if line.strip():
            print(json.dumps(handle_request(json.loads(line))))
            sys.stdout.flush()
