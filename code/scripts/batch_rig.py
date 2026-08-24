"""Batch-rig all official T-pose GLBs (server side).

python batch_rig.py [outdir] — runs stage1_real.py for every glb in ~/work/glbs.
"""

import os
import subprocess
import sys
import time

WORK = os.path.expanduser("~/work")
CODE = os.path.join(WORK, "code")
GLBDIR = os.path.join(WORK, "glbs")
OUTDIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(WORK, "outputs")
os.makedirs(OUTDIR, exist_ok=True)

t0 = time.time()
for name in sorted(os.listdir(GLBDIR)):
    if not name.endswith(".glb"):
        continue
    path = os.path.join(GLBDIR, name)
    out = os.path.join(OUTDIR, name.replace(".glb", "_rigged.glb"))
    if os.path.exists(out) and os.path.getsize(out) > 10_000_000:
        print(f"skip {name} (exists)", flush=True)
        continue
    # wait for a concurrent upload to finish: size must be stable
    s1 = os.path.getsize(path)
    time.sleep(10)
    s2 = os.path.getsize(path)
    if s1 != s2:
        print(f"skip {name} (still uploading, {s2} bytes)", flush=True)
        continue
    print(f"== {name} ({s2} bytes)", flush=True)
    subprocess.run(
        ["python", "-u", "scripts/stage1_real.py", "--glb",
         os.path.join(GLBDIR, name), "--out", out],
        cwd=CODE, check=False,
    )
print(f"batch done in {time.time()-t0:.0f}s")
