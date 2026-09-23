#!/usr/bin/env python3
"""gguf_tensor_diff.py - Test D's instrument, 2026-09-23. Reads only.
Compares two GGUF files tensor by tensor (and metadata key by key), and for every tensor that differs reports
how many elements differ, the largest difference, and how many of the differing elements were EXACTLY ZERO in a
third file (the input model). Answers: did the two arms' weights differ, where, and by how much.
Usage: gguf_tensor_diff.py A.gguf B.gguf INPUT.gguf"""
import sys
sys.path.insert(0, r"M:\llama_cpp_repro_e6ab7c1a\llama.cpp-e6ab7c1a41054a888ada952eab4c886444c2f5ad\gguf-py")
import numpy as np
from gguf import GGUFReader

a, b, inp = (GGUFReader(p) for p in sys.argv[1:4])

def kv(r):
    out = {}
    for name, f in r.fields.items():
        if name.startswith("GGUF."):
            continue
        out[name] = [bytes(p.tobytes()) for p in f.parts]
    return out

ka, kb = kv(a), kv(b)
kdiff = sorted(k for k in set(ka) | set(kb) if ka.get(k) != kb.get(k))
print("metadata keys: %d vs %d | differing: %s" % (len(ka), len(kb), kdiff or "none"))

ta = {t.name: t for t in a.tensors}; tb = {t.name: t for t in b.tensors}; ti = {t.name: t for t in inp.tensors}
# the finetune saver re-serializes (its own key set and tensor order), so tensors are matched by NAME
assert set(ta) == set(tb), "tensor name sets differ: %s" % sorted(set(ta) ^ set(tb))[:10]
print("tensor order identical: %s" % (list(ta) == list(tb)))
ndiff = 0; total = 0
for name, t in ta.items():
    x = np.asarray(t.data).reshape(-1); y = np.asarray(tb[name].data).reshape(-1)
    total += x.size
    if x.dtype != y.dtype or x.shape != y.shape:
        print("%s: type or shape differs" % name); ndiff += 1; continue
    d = x != y
    n = int(d.sum())
    if n == 0:
        continue
    ndiff += 1
    z = np.asarray(ti[name].data).reshape(-1)
    zero_in = int((z[d] == 0).sum()) if z.shape == x.shape else -1
    a_vs_in = int((x != z).sum()) if z.shape == x.shape else -1
    b_vs_in = int((y != z).sum()) if z.shape == x.shape else -1
    print("%-28s %s differs at %d of %d | max|a-b| %.3e | of those, exactly 0 in the input: %d | a vs input: %d, b vs input: %d"
          % (name, x.dtype, n, x.size, float(np.max(np.abs(x[d].astype(np.float64) - y[d].astype(np.float64)))),
             zero_in, a_vs_in, b_vs_in))
print("tensors: %d, differing: %d, elements: %d" % (len(ta), ndiff, total))
