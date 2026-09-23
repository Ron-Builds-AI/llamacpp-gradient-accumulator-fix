#!/usr/bin/env python3
"""make_nozero_model.py - Test E's input, 2026-09-23. PRIVATE throwaway, never shipped, never published.
Copies the Qwen F32 model and replaces every exactly-zero float32 weight with 1e-20. The float32 spacing near
1e-20 is about 1.2e-27, so an AdamW step of about 1e-30 rounds away on every weight: at lr 1e-30 nothing in the
model can move, in either arm. Prints how many weights were replaced and re-reads the file to prove none is 0.
Usage: make_nozero_model.py SRC.gguf DST.gguf"""
import shutil, sys
sys.path.insert(0, r"M:\llama_cpp_repro_e6ab7c1a\llama.cpp-e6ab7c1a41054a888ada952eab4c886444c2f5ad\gguf-py")
import numpy as np
from gguf import GGUFReader

src, dst = sys.argv[1], sys.argv[2]
shutil.copyfile(src, dst)
r = GGUFReader(dst, "r+")
replaced = 0; tensors = 0
for t in r.tensors:
    d = t.data
    if d.dtype != np.float32:
        continue
    z = d == 0
    n = int(z.sum())
    if n:
        d[z] = np.float32(1e-20)
        tensors += 1; replaced += n
r.data.flush()
del r
chk = GGUFReader(dst)
left = sum(int((t.data == 0).sum()) for t in chk.tensors if t.data.dtype == np.float32)
other = sorted({str(t.data.dtype) for t in chk.tensors if t.data.dtype != np.float32})
print("replaced %d exactly-zero weights in %d tensors with 1e-20 | zeros left after re-read: %d | non-f32 tensors: %s"
      % (replaced, tensors, left, other or "none"))
assert left == 0
