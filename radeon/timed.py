"""timed.py OUT.json CMD... : run CMD, record wall seconds, exit code and peak child RSS."""

import json
import resource
import subprocess
import sys
import time

t = time.time()
rc = subprocess.call(sys.argv[2:])
wall = time.time() - t
rss_gib = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**20
with open(sys.argv[1], "w") as f:
    json.dump({"cmd": sys.argv[2:], "returncode": rc, "wall_s": round(wall, 1), "peak_rss_gib": round(rss_gib, 2)}, f, indent=1)
print(f"TIMED rc={rc} wall={wall:.1f}s peak_rss={rss_gib:.2f}GiB")
sys.exit(rc)
