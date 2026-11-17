"""
Train all variants one after another and survive GPU stalls.

Each variant runs in its own process. If its heartbeat file (written every 50
iterations) stops changing for STALL_S seconds, the process is killed and
restarted; train.py resumes from checkpoints/<variant>_last.pt.

    python run_all.py --iters 4000
"""
import argparse
import os
import subprocess
import sys
import time

ORDER = ["unet_l1", "pix2pix", "lee_pix2pix", "ours_nodesp", "ours_nophys", "ours"]
STALL_S = 600


def mtime(p):
    return os.path.getmtime(p) if os.path.exists(p) else None


def run_variant(v, iters):
    hb = f"runs/{v}/heartbeat"
    while not os.path.exists(f"runs/{v}/done"):
        print(f"[run_all] starting {v}", flush=True)
        log = open(f"runs/{v}.stdout.log", "a")
        p = subprocess.Popen([sys.executable, "-u", "train.py", "--variant", v, "--iters", str(iters)],
                             stdout=log, stderr=subprocess.STDOUT)
        last_change, last_m = time.time(), mtime(hb)
        while p.poll() is None:
            time.sleep(20)
            m = mtime(hb)
            if m != last_m:
                last_m, last_change = m, time.time()
            elif time.time() - last_change > STALL_S:
                print(f"[run_all] {v}: no progress for {STALL_S}s -> restarting from last checkpoint", flush=True)
                p.kill(); p.wait()
                break
        log.close()
        if p.returncode not in (0, None, -9, 1) and not os.path.exists(f"runs/{v}/done"):
            print(f"[run_all] {v} exited with code {p.returncode}; retrying", flush=True)
        time.sleep(5)
    print(f"[run_all] finished {v}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    os.makedirs("runs", exist_ok=True)
    for v in (a.only or ORDER):
        os.makedirs(f"runs/{v}", exist_ok=True)
        run_variant(v, a.iters)
    print("[run_all] ALL DONE", flush=True)
