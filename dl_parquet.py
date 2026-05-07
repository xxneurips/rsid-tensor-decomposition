"""Download HF imagenet-1k parquet shards only (no JPEG extraction)."""
import os, sys, time
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
from huggingface_hub import hf_hub_download, list_repo_files

REPO = "benjamin-paine/imagenet-1k"
OUT = "/workspace/imagenet_parquet"
os.makedirs(OUT, exist_ok=True)
TOKEN = os.environ['HF_TOKEN']

files = [f for f in list_repo_files(REPO, repo_type="dataset", token=TOKEN)
         if f.startswith('data/') and f.endswith('.parquet')]
print(f"Total parquet shards: {len(files)}", flush=True)
for i, f in enumerate(files):
    t0 = time.time()
    p = hf_hub_download(REPO, f, repo_type="dataset", token=TOKEN, local_dir=OUT)
    print(f"  [{i+1}/{len(files)}] {f} {os.path.getsize(p)/1e6:.0f}MB elapsed={time.time()-t0:.1f}s", flush=True)
print("DONE")
