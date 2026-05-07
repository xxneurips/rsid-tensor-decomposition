"""
Streaming download of ImageNet-1k from HF (benjamin-paine/imagenet-1k mirror, ungated)
into ImageFolder layout. Each shard is downloaded, decoded, deleted to save disk.

Output:
  /workspace/imagenet_data/imagenet/train/<wnid>/<filename>.JPEG
  /workspace/imagenet_data/imagenet/val/<wnid>/<filename>.JPEG

Peak disk usage: ~150 GB (just the materialized JPEGs).
"""
import os, sys, time, glob, traceback, json
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
import urllib.request
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files

HF_TOKEN = os.environ.get('HF_TOKEN')
if not HF_TOKEN:
    raise SystemExit("HF_TOKEN env var required")

ROOT = '/imagenet_data'
RAW_DIR = os.path.join(ROOT, 'imagenet_raw')
OUT_DIR = os.path.join(ROOT, 'imagenet')
WNIDS_FILE = os.path.join(ROOT, 'imagenet_wnids.txt')


def ensure_wnids():
    """Build the canonical PyTorch ImageNet-2012 wnid list at WNIDS_FILE.

    Source: imagenet_class_index.json from the standard deep-learning-models
    mirror (Keras / TF Hub). Format: {"0": ["n01440764", "tench"], ..., "999": [...]}.
    The integer keys match torchvision IMAGENET1K_V1 output ordering, so a
    pretrained ResNet's logits[i] correspond to wnid = wnids[i].
    """
    os.makedirs(ROOT, exist_ok=True)
    if os.path.exists(WNIDS_FILE):
        with open(WNIDS_FILE) as f:
            wnids = [ln.strip() for ln in f if ln.strip()]
        if len(wnids) == 1000:
            print(f"  using existing wnids file: {WNIDS_FILE}", flush=True)
            return wnids
    print("  fetching imagenet_class_index.json", flush=True)
    url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    raw = urllib.request.urlopen(url, timeout=60).read().decode()
    idx = json.loads(raw)
    wnids = [idx[str(i)][0] for i in range(1000)]
    assert len(wnids) == 1000 and all(w.startswith('n') for w in wnids)
    with open(WNIDS_FILE, 'w') as f:
        f.write('\n'.join(wnids))
    print(f"  wrote {WNIDS_FILE}", flush=True)
    return wnids


def materialize_shard(pq_path, split, wnids):
    """Decode a parquet shard's images into <OUT>/<split>/<wnid>/<file>.JPEG."""
    tbl = pq.read_table(pq_path)
    n = tbl.num_rows
    images = tbl.column('image')
    labels = tbl.column('label')
    cnt = 0
    for i in range(n):
        lbl = labels[i].as_py()
        img = images[i].as_py()
        wnid = wnids[lbl]
        path = img.get('path') or f"{split}_{lbl:04d}_{i:06d}.JPEG"
        fname = os.path.basename(path) or f"{split}_{lbl:04d}_{i:06d}.JPEG"
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            fname = fname + '.JPEG'
        out_dir = os.path.join(OUT_DIR, split, wnid)
        out_path = os.path.join(out_dir, fname)
        if os.path.exists(out_path):
            continue
        with open(out_path, 'wb') as f:
            f.write(img['bytes'])
        cnt += 1
    return n, cnt


def download_and_materialize(filename, split, wnids):
    """Download one parquet shard, extract images, delete parquet.

    Hard 300s timeout per file via signal-based interrupt; up to 3 retries
    with backoff. If it still fails, the caller logs and moves on.
    """
    import signal
    last_err = None
    for attempt in range(3):
        try:
            class TimeoutErr(Exception): pass
            def alarm(*_): raise TimeoutErr("hf_hub_download stuck")
            signal.signal(signal.SIGALRM, alarm)
            signal.alarm(300)
            try:
                pq_path = hf_hub_download(
                    repo_id='benjamin-paine/imagenet-1k', repo_type='dataset',
                    filename=filename, local_dir=RAW_DIR, token=HF_TOKEN,
                )
            finally:
                signal.alarm(0)
            n, cnt = materialize_shard(pq_path, split, wnids)
            try: os.remove(pq_path)
            except OSError: pass
            return n, cnt
        except Exception as e:
            last_err = e
            print(f"    attempt {attempt+1}/3 failed: {type(e).__name__}: {str(e)[:120]}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Gave up on {filename}: {last_err}")


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"=== Resolving canonical wnid list ===", flush=True)
    wnids = ensure_wnids()
    print(f"  {len(wnids)} wnids; first 3: {wnids[:3]}", flush=True)

    print(f"=== Listing dataset files on HF ===", flush=True)
    files = sorted(list_repo_files('benjamin-paine/imagenet-1k', repo_type='dataset', token=HF_TOKEN))
    train_files = [f for f in files if f.startswith('data/train-') and f.endswith('.parquet')]
    val_files = [f for f in files if f.startswith('data/validation-') and f.endswith('.parquet')]
    print(f"  train shards: {len(train_files)}, val shards: {len(val_files)}", flush=True)

    for w in wnids:
        os.makedirs(os.path.join(OUT_DIR, 'train', w), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR, 'val', w), exist_ok=True)

    print(f"=== Streaming val shards ===", flush=True)
    t0 = time.time()
    total_val = 0
    for i, vf in enumerate(val_files):
        try:
            n, cnt = download_and_materialize(vf, 'val', wnids)
            total_val += cnt
            print(f"  [{i+1}/{len(val_files)}] {vf}: rows={n} new={cnt} total={total_val}  elapsed={time.time()-t0:.1f}s", flush=True)
        except Exception as e:
            print(f"  FAILED {vf}: {e}", flush=True)
            traceback.print_exc()
    print(f"  val done: {total_val} new files in {time.time()-t0:.1f}s", flush=True)

    ckpt = os.path.join(OUT_DIR, '.completed_train_shards.txt')
    completed = set()
    if os.path.exists(ckpt):
        with open(ckpt) as f:
            completed = {ln.strip() for ln in f if ln.strip()}
    pending = [tf for tf in train_files if tf not in completed]
    print(f"=== Streaming {len(pending)} train shards (resuming, {len(completed)} already done) ===", flush=True)
    t1 = time.time()
    total_train = 0
    for i, tf in enumerate(pending):
        try:
            n, cnt = download_and_materialize(tf, 'train', wnids)
            total_train += cnt
            with open(ckpt, 'a') as f:
                f.write(tf + '\n')
            if i % 5 == 0 or i == len(pending) - 1:
                print(f"  [{i+1}/{len(pending)}] {tf}: rows={n} new={cnt} total={total_train}  elapsed={time.time()-t1:.1f}s", flush=True)
        except Exception as e:
            print(f"  FAILED {tf}: {e}", flush=True)
            traceback.print_exc()
    print(f"  train done: {total_train} new files in {time.time()-t1:.1f}s", flush=True)

    n_train_classes = sum(1 for d in os.listdir(os.path.join(OUT_DIR, 'train')) if len(os.listdir(os.path.join(OUT_DIR, 'train', d))) > 0)
    n_val_classes = sum(1 for d in os.listdir(os.path.join(OUT_DIR, 'val')) if len(os.listdir(os.path.join(OUT_DIR, 'val', d))) > 0)
    print(f"\n=== summary ===")
    print(f"populated train classes: {n_train_classes}/1000")
    print(f"populated val classes:   {n_val_classes}/1000")
    print(f"total time: {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()
