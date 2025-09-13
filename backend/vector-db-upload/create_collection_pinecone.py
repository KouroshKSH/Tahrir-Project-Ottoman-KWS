#!/usr/bin/env python3
import os
import sys
import json
import time
import argparse
import traceback
import gc
from typing import List, Dict, Tuple
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

import numpy as np
import faiss

from pinecone import Pinecone, ServerlessSpec
import uuid


# ─── FILES ────────────────────────────────────────────────────────────────
INDEX_PATH      = "vector_db.index"
META_PATH       = "vector_db.meta"     # Pickle format
CHECKPOINT_PATH  = "ingest_checkpoint.json"

# ─── PINECONE ──────────────────────────────────────────────────────────────
INDEX_NAME = "ottoman-kws"
API_KEY = "pcsk_XmY1h_HYJ1DqQRB6EWTDAeXgJZuhQJAGpFANnvHyuBxAbE7oMbTwwkMv95boMD9UvaZMn"

# ─── INGEST SETTINGS ─────────────────────────────────────────────────────
BATCH_SIZE       = 1000            # Maximum batch size for Pinecone
RECREATE         = True          # drop+create when starting from batch 1
TARGET_DATATYPE  = "float32"      # Pinecone uses float32

# Pinecone uses simple vectors (no named vectors needed)
USE_NAMED        = False

# ─── RETRY / SELF-RESTART LOGIC ───────────────────────────────────────────
RETRIES_PER_CALL   = 4           # per upsert call retries (reduced for speed)
BASE_SLEEP_SECONDS = 0.1           # reduced sleep time
MAX_RESTARTS       = 300            # total full-process restarts allowed
RESTART_SLEEP      = 8.0          # sleep before re-exec self
MAX_WORKERS        = 16             # parallel upload threads

# ─── Memory monitoring helpers ────────────────────────────────────────────
def get_memory_usage():
    """Get current memory usage in MB."""
    if not HAS_PSUTIL:
        return 0
    try:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except:
        return 0

def log_memory(stage: str):
    """Log current memory usage at different stages."""
    if HAS_PSUTIL:
        mem_mb = get_memory_usage()
        print(f"💾 Memory usage at {stage}: {mem_mb:.1f} MB")

# ─── IO helpers ───────────────────────────────────────────────────────────
class MetadataIterator:
    """Memory-efficient metadata iterator with chunked loading."""
    def __init__(self, path: str):
        self.path = path
        self.total_count = 0
        self.data = None
        self.chunk_cache = {}
        self.chunk_size = 1000  # Load metadata in chunks of 1000
        self._init_format()
    
    def _init_format(self):
        """Load pickle file metadata."""
        import pickle
        try:
            print(f"Loading pickle file: {self.path}")
            with open(self.path, "rb") as f:
                self.data = pickle.load(f)
            self.total_count = len(self.data)
            print(f"Loaded {self.total_count} metadata entries")
        except Exception as e:
            print(f"Failed to read metadata file: {e}")
            raise
    
    def _load_chunk(self, chunk_idx: int) -> Dict[int, Dict]:
        """Load a chunk of metadata entries from pickle data."""
        if chunk_idx in self.chunk_cache:
            return self.chunk_cache[chunk_idx]
        
        start_idx = chunk_idx * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, self.total_count)
        chunk = {}
        
        # Get data from the already-loaded pickle data
        for i in range(start_idx, end_idx):
            if i in self.data:
                chunk[i] = self.data[i]
        
        # Cache management - keep only 3 chunks in memory
        if len(self.chunk_cache) >= 3:
            oldest_chunk = min(self.chunk_cache.keys())
            del self.chunk_cache[oldest_chunk]
            gc.collect()
        
        self.chunk_cache[chunk_idx] = chunk
        return chunk
    
    def get_entry(self, idx: int) -> Dict:
        """Get a single metadata entry by index."""
        if idx >= self.total_count:
            return {}
        
        chunk_idx = idx // self.chunk_size
        chunk = self._load_chunk(chunk_idx)
        return chunk.get(idx, {})

def load_metadata(path: str) -> MetadataIterator:
    """Return metadata iterator for pickle file."""
    if os.path.exists(path):
        return MetadataIterator(path)
    else:
        raise FileNotFoundError(f"Metadata file not found: {path}")

# ─── Vector dtype helpers ─────────────────────────────────────────────────
# Pinecone always uses float32, so we convert accordingly
def to_dtype(vec: np.ndarray, kind: str) -> np.ndarray:
    """Convert a single vector to float32 for Pinecone."""
    return vec.astype(np.float32, copy=False)

# ─── Checkpointing ────────────────────────────────────────────────────────
def save_checkpoint(batch_ix: int, sent_total: int, restarts_left: int):
    payload = {
        "index": INDEX_NAME,
        "batch_ix_next": batch_ix,
        "sent_total": sent_total,
        "restarts_left": restarts_left,
        "ts": time.time(),
    }
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"💾 checkpoint saved ⇒ {payload}")

def load_checkpoint() -> Tuple[int,int,int]:
    if not os.path.exists(CHECKPOINT_PATH):
        return (1, 0, MAX_RESTARTS)
    try:
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            j = json.load(f)
        if j.get("index") != INDEX_NAME:
            # ignore old checkpoint from another index
            return (1, 0, MAX_RESTARTS)
        return (int(j.get("batch_ix_next", 1)),
                int(j.get("sent_total", 0)),
                int(j.get("restarts_left", MAX_RESTARTS)))
    except Exception:
        return (1, 0, MAX_RESTARTS)

def clear_checkpoint():
    try:
        if os.path.exists(CHECKPOINT_PATH):
            os.remove(CHECKPOINT_PATH)
            print("🧹 removed existing checkpoint")
    except Exception:
        pass

# ─── Self-restart ─────────────────────────────────────────────────────────
def restart_self(start_batch: int, restarts_left: int, args_ns):
    if restarts_left <= 0:
        print("⛔ No restarts left; giving up.")
        sys.exit(2)
    save_checkpoint(start_batch, args_ns._sent_total, restarts_left-1)
    print(f"🔁 restarting self in {RESTART_SLEEP:.1f}s (restarts left: {restarts_left-1}) …")
    time.sleep(RESTART_SLEEP)
    py = sys.executable
    script = os.path.abspath(__file__)
    # rebuild argv
    argv = [py, script,
            "--start-batch", str(start_batch),
            "--datatype", args_ns.datatype]
    if args_ns.recreate:
        argv.append("--recreate")
    if args_ns.ignore_checkpoint:
        argv.append("--ignore-checkpoint")
    os.execv(py, argv)

# ─── Robust upsert with retries and batch-splitting ───────────────────────
def upsert_with_retries(index, vectors: List[Dict], namespace: str = "", depth=0) -> int:
    """Try upsert; on failure retry with backoff; if still failing, split batch or skip individual vectors.
    Returns: number of successfully uploaded vectors."""
    size = len(vectors)
    try:
        index.upsert(vectors=vectors, namespace=namespace, async_req=False)
        return size
    except Exception as e:
        msg = str(e)
        print(f"❌ upsert failed (size={size}, depth={depth}): {msg}")
        
        # Check for zero vector error - skip the entire batch
        if "contains only zeros" in msg or "must contain at least one non-zero value" in msg:
            print(f"   ⚠️  Detected zero vector error - skipping entire batch of {size} vectors")
            return 0
        
        # Check if error message contains specific vector IDs that failed
        if "id" in msg.lower() and size == 1:
            # Single vector failed - skip it
            vector_id = vectors[0].get("id", "unknown")
            print(f"   ⚠️  Skipping problematic vector ID: {vector_id}")
            return 0
        
        # try a few timed retries
        for i in range(RETRIES_PER_CALL):
            delay = BASE_SLEEP_SECONDS * (2 ** i)
            print(f"   ↻ retry {i+1}/{RETRIES_PER_CALL} after {delay:.1f}s …")
            time.sleep(delay)
            try:
                index.upsert(vectors=vectors, namespace=namespace, async_req=False)
                print("   ✅ retry succeeded")
                return size
            except Exception as e2:
                print(f"   ✗ retry error: {e2}")

        # Still failing — split if possible
        if size > 1:  # Split even pairs to isolate problematic vectors
            mid = size // 2
            left = vectors[:mid]
            right = vectors[mid:]
            print(f"   ✂ splitting batch: {len(left)} + {len(right)}")
            left_count = upsert_with_retries(index, left, namespace=namespace, depth=depth+1)
            right_count = upsert_with_retries(index, right, namespace=namespace, depth=depth+1)
            return left_count + right_count

        # Single vector still failing - skip it
        vector_id = vectors[0].get("id", "unknown")
        print(f"   ⚠️  Skipping problematic vector ID: {vector_id} after all retries")
        return 0

# ─── main ───────────────────────────────────────────────────────────
def main(start_batch: int, args_ns):
    
    print(f"Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    print(f"Total RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    assert API_KEY and API_KEY.startswith("pcsk_"), "Set valid Pinecone API key"

    # 1) Connect to Pinecone
    print(f"🔗 Pinecone: Connecting...")
    pc = Pinecone(api_key=API_KEY)
    index = pc.Index(INDEX_NAME)

    # 2) Load FAISS
    print(f"🔍 Loading FAISS: {INDEX_PATH}")
    faiss_index = faiss.read_index(INDEX_PATH)
    base = faiss_index.index if hasattr(faiss_index, "index") else faiss_index  # unwrap IDMap if present
    ntotal, dim = faiss_index.ntotal, faiss_index.d
    print(f"🔍 FAISS: {ntotal} vectors × dim {dim}")

    # 3) Check/clear index (only if starting from batch 1)
    if args_ns.recreate and start_batch <= 1:
        clear_checkpoint()
        try:
            # Check if index exists and get stats
            stats = index.describe_index_stats()
            if stats.total_vector_count > 0:
                print(f"🗑️  Index '{INDEX_NAME}' exists with {stats.total_vector_count} vectors")
                print("🧹 Clearing existing vectors...")
                
                # Delete all vectors using delete_all or namespace deletion
                print("⚠️  Warning: Deleting all existing vectors. This may take a while...")
                
                try:
                    # Try to delete all vectors at once (most efficient)
                    index.delete(delete_all=True)
                    print("🗑️  Deleted all vectors using delete_all")
                except Exception as e:
                    print(f"⚠️  delete_all failed: {e}")
                    print("🔄 Falling back to batch deletion...")
                    
                    # Fallback: Get all vector IDs in multiple queries and delete them
                    deleted_count = 0
                    dummy_vector = [0.0] * dim
                    
                    while True:
                        # Query for more vectors
                        matches = index.query(
                            vector=dummy_vector,
                            top_k=10000,  # Maximum allowed per query
                            include_values=False,
                            include_metadata=False
                        )
                        
                        if not matches.matches:
                            break  # No more vectors to delete
                        
                        vector_ids = [match.id for match in matches.matches]
                        print(f"🗑️  Found {len(vector_ids)} more vectors to delete...")
                        
                        # Delete in batches of 1000 (Pinecone limit)
                        batch_size = 1000
                        for i in range(0, len(vector_ids), batch_size):
                            batch_ids = vector_ids[i:i + batch_size]
                            index.delete(ids=batch_ids)
                            deleted_count += len(batch_ids)
                            print(f"   Deleted batch of {len(batch_ids)} vectors (total deleted: {deleted_count})")
                        
                        # Wait a bit for deletions to propagate before next query
                        time.sleep(1)
                
                # Verify deletion
                time.sleep(3)  # Wait for deletion to propagate
                final_stats = index.describe_index_stats()
                print(f"✅ Index cleared. Remaining vectors: {final_stats.total_vector_count}")
            else:
                print(f"✅ Index '{INDEX_NAME}' is already empty")
        except Exception as e:
            print(f"❌ Error clearing index: {e}")
            print(f"ℹ️  Make sure index '{INDEX_NAME}' exists in Pinecone console")
            print(f"ℹ️  If you have many vectors, consider deleting the index manually and recreating it")
            sys.exit(1)
    else:
        print("ℹ️  Re-using existing index (no recreate)")

    # 4) Metadata
    log_memory("before metadata loading")
    meta = load_metadata(META_PATH)
    log_memory("after metadata loading")
    if meta.total_count != ntotal:
        print(f"⚠️  metadata count ({meta.total_count}) != vectors ({ntotal}) — continuing")

    # 5) Upsert in batches
    num_batches = (ntotal + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"📤 Upserting in {num_batches} batches (≤{BATCH_SIZE})")

    # honor checkpoint (unless ignored)
    sent = args_ns._sent_total  # from checkpoint loader
    if start_batch > 1 and sent == 0:
        sent = min((start_batch - 1) * BATCH_SIZE, ntotal)
    if sent:
        print(f"ℹ️  Assuming first {sent} vectors already stored (start_batch={start_batch})")

    for b_ix, start in enumerate(range(0, ntotal, BATCH_SIZE), start=1):
        if b_ix < start_batch:
            continue

        end = min(ntotal, start + BATCH_SIZE)
        size = end - start

        # 5a) reconstruct block
        log_memory(f"batch {b_ix} start")
        try:
            V = base.reconstruct_n(start, size)  # (size, dim) float32
            if not isinstance(V, np.ndarray):
                V = np.array(V, dtype=np.float32)
            if V.shape != (size, dim):
                raise RuntimeError(f"reconstruct_n returned {V.shape}, expected {(size, dim)}")
            log_memory(f"batch {b_ix} after reconstruct")
        except Exception as e:
            print("❌ reconstruct_n failed:", e)
            traceback.print_exc()
            # Fatal — checkpoint and restart
            restart_self(b_ix, args_ns._restarts_left, args_ns)
            return  # never reached

        # 5b) convert each row to target dtype (float32 for Pinecone)
        rows = [to_dtype(V[i], args_ns.datatype or TARGET_DATATYPE) for i in range(size)]
        V_conv = np.stack(rows, axis=0)
        # Clear original vector to free memory
        del V, rows

        # 5c) build Pinecone vector list
        vectors = []
        empty_metadata_count = 0
        for off in range(size):
            idx = start + off
            entry = meta.get_entry(idx) if idx < meta.total_count else {}
            
            # Validate that we have essential metadata
            if not entry or (not entry.get("document") and not entry.get("doc")) or not entry.get("id"):
                print(f"❌ WARNING: Vector {idx} missing essential metadata - entry: {entry}")
                empty_metadata_count += 1
                # Skip this vector entirely rather than upload with empty metadata
                continue
            
            metadata = {
                "doc":    entry.get("document") or entry.get("doc"),
                "patch":  entry.get("patch") or entry.get("id"),
                "coords": entry.get("coords"),
            }
            # Remove None values from metadata and convert to strings (Pinecone requirement)
            metadata = {k: str(v) for k, v in metadata.items() if v is not None}
            
            # Final validation - ensure metadata is not empty
            if not metadata:
                print(f"❌ WARNING: Vector {idx} has empty metadata after processing - skipping")
                empty_metadata_count += 1
                continue
            
            vec_field = V_conv[off].astype(np.float32).tolist()  # Pinecone needs float32
            vectors.append({
                "id": str(idx),  # Pinecone IDs must be strings
                "values": vec_field,
                "metadata": metadata
            })
        
        if empty_metadata_count > 0:
            print(f"⚠️  Skipped {empty_metadata_count} vectors with missing/empty metadata in this batch")

        # Skip this entire batch if no vectors have valid metadata
        if not vectors:
            print(f"   Skipping batch {b_ix} entirely - no vectors with valid metadata")
            # Still count as "processed" and move to next batch
            sent += size  # Add the original batch size to maintain index alignment
            print(f"[{b_ix}/{num_batches}] ⚠️  batch skipped (total processed: {sent})", flush=True)
            # Clear processed data and continue to next batch
            del V_conv
            gc.collect()
            save_checkpoint(b_ix + 1, sent, args_ns._restarts_left)
            continue

        # 5d) robust upsert for this batch (async)
        try:
            uploaded_count = upsert_with_retries(index, vectors)
            skipped_count = len(vectors) - uploaded_count
            if skipped_count > 0:
                print(f"   ⚠️  Skipped {skipped_count} problematic vectors in this batch")
        except Exception as e:
            print(f"❌ batch {b_ix} final failure: {e}")
            # Save checkpoint for this batch to retry upon restart
            save_checkpoint(b_ix, sent, args_ns._restarts_left)
            # Full process restart
            restart_self(b_ix, args_ns._restarts_left, args_ns)
            return  # never reached

        sent += uploaded_count
        print(f"[{b_ix}/{num_batches}] ✅ upserted {uploaded_count}/{len(vectors)} (total {sent})", flush=True)

        # Clear processed data to free memory
        del vectors, V_conv
        
        # Force garbage collection after each batch
        gc.collect()

        # after a SUCCESSFUL batch: save checkpoint for the *next* batch
        save_checkpoint(b_ix + 1, sent, args_ns._restarts_left)

    # 6) verify: count + sanity search
    stats = index.describe_index_stats()
    cnt = stats.total_vector_count
    print(f"🔢 count: {cnt}")
    
    # Simple sanity search with a random vector
    if cnt > 0:
        print("🔎 Performing sanity search...")
        # Create a random query vector of correct dimension
        query_vector = np.random.randn(dim).astype(np.float32).tolist()
        hits = index.query(vector=query_vector, top_k=3, include_metadata=True)
        if hits.matches:
            print(f"🔍 Sanity search returned {len(hits.matches)} results")
            for match in hits.matches[:2]:  # Show first 2 results
                print(f"   ID: {match.id}, Score: {match.score:.4f}")
        else:
            print("⚠️  Sanity search returned no results")

    print("✅ Ingest complete.")
    # success: clear checkpoint
    clear_checkpoint()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-batch", type=int, default=None, help="1-based batch to start from")
    ap.add_argument("--datatype", type=str, default=TARGET_DATATYPE,
                    help="float32 (Pinecone uses float32)")
    ap.add_argument("--recreate", action="store_true", help="force check index on start")
    ap.add_argument("--ignore-checkpoint", action="store_true",
                    help="do not auto-resume from ingest_checkpoint.json")
    args = ap.parse_args()

    # Load/merge checkpoint
    ckpt_batch, ckpt_sent, ckpt_restarts = load_checkpoint()
    if args.ignore_checkpoint:
        start_batch = args.start_batch if args.start_batch is not None else 1
        restarts_left = MAX_RESTARTS
        sent_total = 0
    else:
        # If start-batch not specified, use checkpoint; otherwise use CLI argument
        if args.start_batch is None:
            start_batch = ckpt_batch  # Will be 1 if no checkpoint exists
        else:
            start_batch = args.start_batch
        restarts_left = ckpt_restarts
        sent_total = ckpt_sent if start_batch >= ckpt_batch else 0

    # attach runtime fields to namespace (for self-restart)
    args._restarts_left = restarts_left
    args._sent_total = sent_total

    try:
        main(start_batch, args)
    except SystemExit:
        raise
    except Exception as e:
        # Fatal at top level — checkpoint and restart if possible
        print(f"💥 Top-level failure: {e}")
        traceback.print_exc()
        save_checkpoint(start_batch, sent_total, restarts_left)
        # If something really odd happened before we set args._restarts_left,
        # default to one attempt.
        if not hasattr(args, "_restarts_left"):
            args._restarts_left = 1
        restart_self(start_batch, args._restarts_left, args)