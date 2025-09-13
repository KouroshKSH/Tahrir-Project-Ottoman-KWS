# ---- SQLite shim for serverless runtimes (no system libsqlite) ----
import sys
try:
    import pysqlite3 as sqlite3
    sys.modules["sqlite3"] = sqlite3
except Exception:
    pass

import os, io, hashlib, time, math, json, traceback, urllib.parse, random as _rnd
from typing import List, Dict, Any, Tuple

import functions_framework
from flask import jsonify, make_response

from PIL import Image
import torch
from torchvision import transforms

import boto3
from botocore.config import Config
from pinecone import Pinecone, ServerlessSpec

# ==================== ENV ====================
DEBUG_NO_MODEL       = os.getenv("DEBUG_NO_MODEL", "0") == "1"
ASSUME_UNIT_NORM     = os.getenv("ASSUME_UNIT_NORM", "1") == "1"  # treat dot like cosine

AWS_REGION           = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET            = os.environ.get("S3_BUCKET", "")
S3_KEY               = os.environ.get("S3_KEY", "")

AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_SESSION_TOKEN     = os.getenv("AWS_SESSION_TOKEN", "")

MODEL_PATH          = os.getenv("MODEL_PATH", "/tmp/embedder.pth")
MODEL_SHA256        = os.getenv("MODEL_SHA256", "")
MODEL_FORMAT        = os.getenv("MODEL_FORMAT", "jit")

INPUT_SIZE          = int(os.getenv("INPUT_SIZE", "224"))
NORM_MEAN           = [0.485, 0.456, 0.406]
NORM_STD            = [0.229, 0.224, 0.225]

# Pinecone Configuration
PINECONE_API_KEY    = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")

MAX_CANDIDATES      = int(os.getenv("MAX_CANDIDATES", "10000"))  # Increased default to get more candidates
DEFAULT_THRESHOLD   = float(os.getenv("DEFAULT_THRESHOLD", "0.5"))  # More permissive default

# Pager knobs for robustness
PAGE_SIZE_DEFAULT   = int(os.getenv("PAGE_SIZE", "64"))
RETRY_MAX           = int(os.getenv("RETRY_MAX", "4"))
BASE_BACKOFF        = float(os.getenv("BASE_BACKOFF", "0.75"))
PAGE_DELAY_MS       = int(os.getenv("PAGE_DELAY_MS", "0"))

ALLOWED             = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,https://ottoman-kws.web.app").split(",")

torch.set_num_threads(max(1, int(os.getenv("TORCH_THREADS", "1"))))
_model  = None
_pre    = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])

BUILD_ID = os.getenv("BUILD_ID", f"b{int(time.time())}")
ROUTE_VERSION = "match_v16_pinecone_threshold"

# Initialize Pinecone
_pc = None
_index = None

# ==================== CORS ====================
def _cors(resp, origin: str):
    allow = origin if ("*" in ALLOWED or origin in ALLOWED) else (ALLOWED[0] if ALLOWED else "*")
    resp.headers["Access-Control-Allow-Origin"] = allow
    resp.headers["Vary"] = "Origin"
    resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    resp.headers["Access-Control-Max-Age"] = "3600"
    return resp

# ==================== MODEL ====================
def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def _download_model_from_s3():
    if os.path.exists(MODEL_PATH) or not S3_BUCKET or not S3_KEY:
        return
    session_kwargs = {"region_name": AWS_REGION}
    if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
        session_kwargs.update(
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        )
        if AWS_SESSION_TOKEN:
            session_kwargs["aws_session_token"] = AWS_SESSION_TOKEN
    s3 = boto3.client("s3", config=Config(signature_version="s3v4"), **session_kwargs)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        s3.download_fileobj(S3_BUCKET, S3_KEY, f)
    if MODEL_SHA256 and _sha256(MODEL_PATH) != MODEL_SHA256:
        raise RuntimeError("Model checksum mismatch")

def _file_info(path: str):
    try: sz = os.path.getsize(path)
    except Exception: sz = -1
    return {"exists": os.path.exists(path), "size": sz, "path": path}

def _build_model_for_state_dict():
    raise NotImplementedError("Provide architecture when MODEL_FORMAT=state_dict")

def _ensure_model():
    global _model
    if _model is not None:
        return _model

    if DEBUG_NO_MODEL:
        class _Dummy:
            def eval(self): return self
            def __call__(self, x):
                import numpy as np, torch as _t
                dim = _VEC_DIM or 128
                vec = np.array([_rnd.random() for _ in range(dim)], dtype="float32")
                return _t.from_numpy(vec).unsqueeze(0)
        _model = _Dummy()
        return _model

    _download_model_from_s3()
    finfo = _file_info(MODEL_PATH)
    ver = getattr(torch, "__version__", "unknown")

    try:
        if MODEL_FORMAT.lower() == "jit":
            _model = torch.jit.load(MODEL_PATH, map_location="cpu").eval()
            return _model
        else:
            import torch.nn as nn
            model = _build_model_for_state_dict()
            state = torch.load(MODEL_PATH, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=False)
            _model = model.eval()
            return _model
    except Exception as e_jit:
        inspect = {"torch_version": ver, "model_format_env": MODEL_FORMAT, "file": finfo}
        try:
            obj = torch.load(MODEL_PATH, map_location="cpu")
            inspect["torch_load_type"] = type(obj).__name__
            if isinstance(obj, dict):
                inspect["torch_dict_keys"] = list(obj.keys())[:15]
        except Exception as e2:
            inspect["torch_load_error"] = str(e2)
        raise RuntimeError("Model load failed; wrong artifact or Torch mismatch: " + json.dumps(inspect)) from e_jit

def _embed(image_bytes: bytes):
    model = _ensure_model()
    if DEBUG_NO_MODEL:
        import numpy as np
        dim = _VEC_DIM or 128
        return np.array([_rnd.random() for _ in range(dim)], dtype="float32").tolist()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = _pre(img).unsqueeze(0)
    with torch.no_grad():
        vec = model(x)
        if isinstance(vec, (list, tuple)):
            vec = vec[0]
        emb = vec.squeeze(0).cpu().numpy().astype("float32").tolist()
    return emb

# ==================== PINECONE ====================
def _init_pinecone():
    global _pc, _index, _VEC_DIM, _METRIC
    if _pc is not None and _index is not None:
        return _pc, _index
    
    _pc = Pinecone(api_key=PINECONE_API_KEY)
    _index = _pc.Index(PINECONE_INDEX_NAME)
    
    # Set defaults - these will be updated when we get actual embeddings or stats
    if _VEC_DIM is None:
        _VEC_DIM = 512
    if _METRIC is None:
        _METRIC = "cosine"
    
    return _pc, _index

def _pinecone_healthcheck():
    try:
        pc, index = _init_pinecone()
        stats = index.describe_index_stats()
        # Convert Pinecone response to serializable dict
        stats_dict = {
            "total_vector_count": stats.total_vector_count if hasattr(stats, 'total_vector_count') else 0,
            "dimension": stats.dimension if hasattr(stats, 'dimension') else 0,
            "index_fullness": stats.index_fullness if hasattr(stats, 'index_fullness') else 0,
        }
        if hasattr(stats, 'namespaces') and stats.namespaces:
            stats_dict["namespaces"] = {k: {"vector_count": v.vector_count} for k, v in stats.namespaces.items()}
        return 200, {"status": "healthy", "stats": stats_dict}
    except Exception as e:
        return 500, {"status": "unhealthy", "error": str(e)}

def _score_to_distance(score, metric="cosine"):
    """Convert Pinecone score to distance."""
    s = float(score)
    if metric == "cosine":
        # Pinecone cosine similarity ranges from -1 to 1
        # Distance = 1 - similarity (ranges 0 to 2)
        return max(0.0, 1.0 - s)
    elif metric == "euclidean":
        # For euclidean, score is already distance
        return s
    elif metric == "dotproduct":
        if ASSUME_UNIT_NORM:
            return max(0.0, 1.0 - s)
        # For non-unit vectors, convert score to approximate distance
        return 1.0 / (1.0 + max(0.0, s))
    return max(0.0, 1.0 - s)

def _passes_distance_threshold(score, metric, threshold):
    """Check if a score passes the distance threshold."""
    distance = _score_to_distance(score, metric)
    return distance <= threshold

def _get_random_vector_pinecone(sample_size: int = 1) -> Tuple[int, Dict]:
    """Get random vectors from Pinecone for diagnostic purposes."""
    try:
        pc, index = _init_pinecone()
        
        # Query with a random vector to get some results
        random_vector = [_rnd.random() for _ in range(_VEC_DIM or 512)]
        
        result = index.query(
            vector=random_vector,
            top_k=sample_size,
            include_metadata=True,
            include_values=True
        )
        
        return 200, {
            "matches": result.get('matches', []),
            "sample_size": len(result.get('matches', []))
        }
    except Exception as e:
        return 500, {"error": str(e)}

def _search_pinecone(
    vector: List[float],
    threshold: float,
    include_metadata: bool = True,
    namespace: str = "",
    filter_dict: Dict = None,
) -> Tuple[List[Dict], Dict]:
    """
    Search Pinecone index and return ALL results that pass the distance threshold.
    """
    try:
        pc, index = _init_pinecone()
        
        # Ensure we have the right vector dimension
        global _VEC_DIM
        if _VEC_DIM != len(vector):
            _VEC_DIM = len(vector)
        
        # Query Pinecone with MAX_CANDIDATES to get a large pool
        query_kwargs = {
            "vector": vector,
            "top_k": MAX_CANDIDATES,
            "include_metadata": include_metadata,
            "include_values": False,
        }
        
        if namespace:
            query_kwargs["namespace"] = namespace
        if filter_dict:
            query_kwargs["filter"] = filter_dict
        
        result = index.query(**query_kwargs)
        
        # Process and filter results by threshold - RETURN ALL THAT PASS
        matches = []
        raw_matches = result.get('matches', [])
        
        for match in raw_matches:
            score = match.get('score', 0.0)
            distance = _score_to_distance(score, _METRIC)
            
            # Apply threshold filter - keep ALL that pass
            if distance <= threshold:
                metadata = match.get('metadata', {})
                
                # Extract document info - adjust these field names based on your metadata structure
                doc = metadata.get('doc', '') or metadata.get('document', '') or metadata.get('text', '')
                coords = metadata.get('coords')
                
                # Handle coordinate extraction from separate fields if needed
                if coords is None:
                    x = metadata.get('x')
                    y = metadata.get('y')
                    w = metadata.get('width')
                    h = metadata.get('height')
                    if all(v is not None for v in [x, y, w, h]):
                        coords = [int(x), int(y), int(x) + int(w), int(y) + int(h)]
                
                matches.append({
                    "id": match.get('id'),
                    "doc": doc,
                    "coords": coords,
                    "score": float(score),
                    "distance": distance,
                    "metadata": metadata,
                })
        
        # Sort by distance (ascending) but DON'T limit by k
        matches.sort(key=lambda x: x["distance"])
        
        search_info = {
            "total_candidates": len(raw_matches),
            "matches_within_threshold": len(matches),
            "threshold_applied": threshold,
            "metric": _METRIC,
        }
        
        return matches, search_info
        
    except Exception as e:
        raise RuntimeError(f"Pinecone search failed: {str(e)}")

def _probe_with_random_vector_pinecone(
    threshold: float,
    namespace: str = "",
) -> Tuple[List[Dict], Dict]:
    """
    Get a random vector from Pinecone and search with it to test connectivity.
    """
    diag = {"phase": "probe_start"}
    
    # Get a random sample vector
    sc, random_data = _get_random_vector_pinecone(1)
    if sc != 200:
        raise RuntimeError(f"Failed to get random vector: {random_data}")
    
    matches = random_data.get("matches", [])
    if not matches:
        raise RuntimeError("No matches found for probe")
    
    sample_match = matches[0]
    if 'values' not in sample_match:
        raise RuntimeError("Sample vector values not available")
    
    vector = sample_match['values']
    diag["random_point"] = {
        "id": sample_match.get('id'),
        "vector_len": len(vector),
    }
    
    # Search with this random vector - return ALL matches within threshold
    diag["phase"] = "probe_search"
    probe_matches, search_info = _search_pinecone(
        vector=vector,
        threshold=threshold,
        namespace=namespace,
    )
    
    diag["phase"] = "probe_complete"
    diag["matches_found"] = len(probe_matches)
    diag["search_info"] = search_info
    
    return probe_matches, diag

def _get_collection_stats_pinecone(namespace: str = "") -> Dict[str, Any]:
    """Get comprehensive statistics about the Pinecone index."""
    try:
        pc, index = _init_pinecone()
        stats = index.describe_index_stats()
        
        # Convert to serializable dict
        index_info = {
            "index_accessible": True,
            "total_vector_count": stats.total_vector_count if hasattr(stats, 'total_vector_count') else 0,
            "dimension": stats.dimension if hasattr(stats, 'dimension') else 0,
            "index_fullness": stats.index_fullness if hasattr(stats, 'index_fullness') else 0,
        }
        
        if hasattr(stats, 'namespaces') and stats.namespaces:
            index_info["namespaces"] = {
                k: {"vector_count": v.vector_count} 
                for k, v in stats.namespaces.items()
            }
        
        # Update global dimension if available
        global _VEC_DIM
        if hasattr(stats, 'dimension') and stats.dimension and _VEC_DIM != stats.dimension:
            _VEC_DIM = stats.dimension
        
        return index_info
        
    except Exception as e:
        return {
            "index_accessible": False,
            "error": str(e)
        }

# ==================== HANDLER ====================
_VEC_DIM = None
_METRIC  = "cosine"  # Pinecone default

@functions_framework.http
def match(request):
    origin = request.headers.get("Origin", "")
    if request.method == "OPTIONS":
        resp = make_response(("", 204))
        resp.headers["X-App-Build"] = BUILD_ID
        resp.headers["X-App-Route"] = ROUTE_VERSION
        return _cors(resp, origin)

    debug = request.args.get("debug") == "1"
    probe_random = request.args.get("probe_random") == "1"
    inspect = request.args.get("inspect") == "1"
    namespace = request.args.get("namespace", "")

    diag: Dict[str, Any] = {"build": BUILD_ID, "route": ROUTE_VERSION, "phase": "start"}
    
    try:
        # Basic environment info
        diag["env"] = {
            "PINECONE_INDEX_NAME": PINECONE_INDEX_NAME,
            "PINECONE_ENVIRONMENT": PINECONE_ENVIRONMENT,
            "ASSUME_UNIT_NORM": ASSUME_UNIT_NORM,
        }

        # Initialize Pinecone
        pc, index = _init_pinecone()
        
        # Health check
        sc, health_info = _pinecone_healthcheck()
        diag["pinecone_health"] = {"status_code": sc, "info": health_info}
        if sc != 200:
            raise RuntimeError(f"Pinecone not healthy: {sc}")

        # Get parameters - ONLY need threshold from frontend
        threshold = float(request.args.get("threshold") or DEFAULT_THRESHOLD)
        
        diag["params"] = {
            "threshold": threshold,
            "metric": _METRIC,
            "namespace": namespace
        }

        # Handle probe_random request
        if probe_random:
            diag["phase"] = "probe_random"
            matches, probe_diag = _probe_with_random_vector_pinecone(
                threshold=threshold,
                namespace=namespace,
            )
            # Merge probe diagnostics
            diag.update(probe_diag)
            
            resp = jsonify({
                "probe_top": matches,
                "message": f"Probe successful: found {len(matches)} matches within threshold {threshold}",
                **({"diag": diag} if debug else {})
            })
            resp.headers["X-App-Build"] = BUILD_ID
            resp.headers["X-App-Route"] = ROUTE_VERSION
            return _cors(resp, origin)

        # Handle inspect request
        if inspect:
            diag["phase"] = "inspect"
            
            # Get collection statistics
            collection_stats = _get_collection_stats_pinecone(namespace)
            diag["collection_stats"] = collection_stats
            
            resp = jsonify({
                "index_info": collection_stats,
                "message": f"Inspection complete: index has {collection_stats.get('total_vector_count', 'unknown')} vectors",
                **({"diag": diag} if debug else {})
            })
            resp.headers["X-App-Build"] = BUILD_ID
            resp.headers["X-App-Route"] = ROUTE_VERSION
            return _cors(resp, origin)

        # Normal search: require image
        if "image" not in request.files:
            raise ValueError("Send multipart/form-data with field 'image' (or use probe_random=1 or inspect=1)")

        image_bytes = request.files["image"].read()
        
        # Generate embedding
        emb = _embed(image_bytes)
        diag["embed_len"] = len(emb)
        
        # Update vector dimension if needed
        global _VEC_DIM
        if _VEC_DIM != len(emb):
            _VEC_DIM = len(emb)
        
        if any((not math.isfinite(v)) for v in emb):
            raise ValueError("Embedding has NaN/Inf")

        # Quick stats to spot scaling/normalization issues
        diag["embed_stats"] = {
            "l2": float(sum(e*e for e in emb)) if emb else 0.0,
            "min": float(min(emb)) if emb else 0.0,
            "max": float(max(emb)) if emb else 0.0,
        }
        
        # Add collection stats for debugging
        diag["collection_stats"] = _get_collection_stats_pinecone(namespace)

        # Search with threshold filtering - RETURN ALL MATCHES
        diag["phase"] = "search_pinecone"
        
        matches, search_info = _search_pinecone(
            vector=emb,
            threshold=threshold,
            namespace=namespace,
        )
        
        diag["search_info"] = search_info

        # If no matches found and debug is enabled, provide diagnostic info
        if len(matches) == 0 and debug:
            diag["search_diagnostic"] = {
                "threshold_used": threshold,
                "embedding_norm": sum(x*x for x in emb) ** 0.5,
                "suggestion": f"Try increasing threshold above {threshold} or check if embeddings are properly normalized"
            }

        resp = jsonify({
            "matches": matches,
            "threshold_applied": threshold,
            "total_matches": len(matches),
            "total_candidates_searched": search_info.get("total_candidates", len(matches)),
            "message": f"Found {len(matches)} matches within distance threshold {threshold}",
            **({"diag": diag} if debug else {})
        })
        resp.headers["X-App-Build"] = BUILD_ID
        resp.headers["X-App-Route"] = ROUTE_VERSION
        return _cors(resp, origin)

    except Exception as e:
        diag["exception"] = str(e)
        diag["trace"] = traceback.format_exc()
        print(json.dumps({"level": "error", "diag": diag}))
        resp = jsonify({"error": str(e), **({"diag": diag} if debug else {})})
        resp.status_code = 400
        resp.headers["X-App-Build"] = BUILD_ID
        resp.headers["X-App-Route"] = ROUTE_VERSION
        return _cors(resp, origin)