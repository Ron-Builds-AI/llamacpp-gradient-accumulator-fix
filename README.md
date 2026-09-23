# Gradient accumulators are never cleared between steps on llama.cpp's dynamic-graph training path

`llama-finetune` trains through the dynamic-graph path in `ggml-opt`, and it is the only tool that
does. On that path the gradient accumulators are zeroed only when their buffer is first allocated, so
**the gradient applied at step *k* is the sum of the gradients from steps 1 through *k*.** This work is
not first, and does not claim to be: srossitto79 found the same defect and proposed a fix for it in
#22704 on 2026-05-05, the earliest report of it we found. It was never merged, and master still carries
the defect. See [Earlier work](#earlier-work-on-the-same-defect).

The accumulation itself raises no error and no warning.

Everything below is checked against master at `e6ab7c1a` (2026-09-22). `ggml/src/ggml-opt.cpp` there
was last changed on 2026-04-08.

## The code path

All in `ggml/src/ggml-opt.cpp` at `e6ab7c1a` unless another file is named. Quotes are verbatim.

| line | code | what it does |
|---|---|---|
| 250 | `/*ctx_compute     =*/ nullptr,` in `ggml_opt_default_params` | no compute context by default |
| 565 | `result->static_graphs = result->ctx_compute;` | so the graphs are dynamic |
| `src/llama-context.cpp`, `opt_init` | sets `opt_period`, never `ctx_compute` | so `llama-finetune` takes the dynamic path |
| 328-329 | `const bool accumulate = opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_GRAD && !(opt_ctx->static_graphs && ...)` | accumulation is on for every gradient build when the graphs are dynamic |
| 458 | `if (opt_ctx->grad_accs.empty()) {` | the accumulators are created once, in `ctx_static`, and every rebuilt graph reuses them |
| 496-499, 540-544 | `ggml_graph_reset(opt_ctx->gb_grad);` and `ggml_graph_reset(opt_ctx->gb_opt);`, each in the branch that first allocates `buf_static` for a training build | they are zeroed when the buffer is first allocated for training, and only then. `llama-finetune` takes 540-544, since its build type is `GGML_OPT_BUILD_TYPE_OPT` (line 254). The forward-only branch at 452-455 allocates without a reset |
| 727-729 | `if (opt_ctx->build_type == GGML_OPT_BUILD_TYPE_OPT && opt_ctx->opt_period > 1 && opt_ctx->opt_i == 0) {` `ggml_graph_reset(opt_ctx->gb_grad);` `}` | the reset meant to clear them at each accumulation window |
| 830 | `opt_ctx->gb_grad              = nullptr;` at the end of every eval when `!static_graphs` | why that reset never works |
| `ggml/src/ggml.c:7611-7613` | `if (!cgraph) {` `return;` `}` | where the null graph returns |

`ggml_opt_reset` (line 596) is never called from llama, and on this path it would be handed the same
null graphs, because lines 830-831 clear both `gb_grad` and `gb_opt` after every eval.

The reset at 727-729 misses for two reasons:

1. It runs before the graph is rebuilt, and after the first step `gb_grad` is null (line 830), so
   `ggml_graph_reset` returns at its null guard. This holds at any `opt_period`.
2. It is gated on `opt_period > 1`, and `opt_period` is `n_batch / n_ubatch`
   (`src/llama-context.cpp:3500`). When `-b` equals `-ub`, as in both reproductions below, it is 1 and
   the gate is closed as well. With the default `-b 2048 -ub 512` (`common/common.h:451-452`) it is 4,
   and the gate is open, but only when the context is at least 2048 tokens, because `n_batch` is capped
   at the context size (`src/llama-context.cpp:246`). Reason 1 is the one that always applies.

## Why the tests do not catch it

On master, every test in `tests/test-opt.cpp` runs on static graphs. The shared helper sets
`opt_params.ctx_compute` (line 143), and the `ggml_opt_fit` cases pass a compute context in, which
`ggml_opt_fit` copies into its parameters (`ggml/src/ggml-opt.cpp:1031`). Nothing exercises the dynamic
path, and that is the only path `llama-finetune` uses.

## The fix

`PATCH_gradient-accumulator-reset.diff`: one hunk in `ggml_opt_alloc`, directly after the dynamic
rebuild.

```c
if (opt_ctx->gb_grad && opt_ctx->opt_i == 0) {
    ggml_graph_reset(opt_ctx->gb_grad);
}
```

- It runs after the rebuild, so `gb_grad` exists.
- It resets `gb_grad`, not `gb_opt`. `ggml_graph_reset` clears AdamW's moments when it meets an
  optimizer step node (`ggml.c:7620-7624`), and those live only in `gb_opt`. `gb_grad` holds the
  forward and backward passes and no optimizer step, so this zeroes the accumulators, sets the loss
  gradient to 1, and leaves the moments alone.
- `opt_i == 0` is the start of an accumulation window. At `opt_period == 1` that is every step. At
  larger periods the micro-batches still accumulate within a window, which is what the accumulators
  are for.

It applies cleanly to master `e6ab7c1a`.

**Regression test.** `PATCH_regression_test.diff` adds `test_grad_dynamic` to `tests/test-opt.cpp`. It
is the dynamic-graph twin of the existing `test_grad`: no `ctx_compute` at init, the forward graph
rebuilt every step and handed over with `ggml_opt_prepare_alloc` as `llama-finetune` does, and
`opt_period` 1. The gradient of the loss with respect to the single weight is exactly 1 at every step,
so the accumulator must read 1 after every step.

| build | `test_grad_dynamic`, AdamW and SGD | the rest of `test-opt` |
|---|---|---|
| master + the test | fails, reading 1 2 3 4 5 6 | passes (118 of 119 and 46 of 47, the one failure being the new test) |
| master + the test + the fix | passes | passes (119 of 119 and 47 of 47) |

Measured on both machines (`evidence/regression_test/`).

## Reproducing it

`PATCH_reproduce.diff`, against `e6ab7c1a`, adds two things, both marked `REPRODUCTION ONLY` in the code:

1. The fix above, with an off switch. `FT_NO_GRAD_RESET`, set to any value, skips the reset.
2. A probe. `FT_GRAD_PROBE` prints, once per optimizer step, the largest absolute gradient and the
   largest absolute weight across all F32 parameters.

**Build**, CPU only: `cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_NATIVE=ON`,
then `cmake --build build --config Release --target llama-finetune`. The Windows builds used the Visual
Studio 2026 generator and `-DGGML_CUDA=OFF`.

### The quick one: stories260K, on master with nothing else changed

stories260K is llama.cpp's own small test model, the one #27199's example trains. Stock master trains
it to the end of the epoch (`evidence/stories260K/*/stock_master.txt`), so this reproduction needs
`PATCH_reproduce.diff` and nothing more. Each run takes a few seconds on a CPU.

**Model:** `https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf`, 1,185,376
bytes, SHA256 `270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d`
(`evidence/stories260K/model_header.txt`).

```
FT_NO_GRAD_RESET=1 FT_GRAD_PROBE=1 llama-finetune \
  -m stories260K.gguf -f REPRO_window_synthetic.txt -o out.gguf \
  -c 128 -b 128 -ub 128 -ngl 0 -fa off -opt adamw -epochs 1 -lr 1e-30 -t 6
```

Then the same command without `FT_NO_GRAD_RESET`, and compare `max|grad|` across steps.

### A larger model: Qwen2.5-0.5B-Instruct

On this model, stock `llama-finetune` aborts before its first step:

```
ggml/src/ggml.c:7291: GGML_ASSERT(cgraph->n_nodes < cgraph->size) failed
```

That is measured on both machines (`evidence/qwen2.5-0.5b-instruct/*/stock_master.txt`), and the same
abort happens with `PATCH_reproduce.diff` applied (`fix_and_probe_only.txt`). The training graph needs
more node slots than the inference budget in `llama_context::graph_max_nodes` provides.
`PATCH_node_budget.diff` raises the default case from 8 to 64 nodes per tensor, and from 1024 to 8192 at
minimum. 64 is the value this was run with, not a measured minimum, and it is not proposed upstream.
The Qwen runs were built with both patches.

**Model:** `qwen2.5-0.5b-instruct-q4_k_m.gguf` from `Qwen/Qwen2.5-0.5B-Instruct-GGUF` (491,400,032 bytes,
SHA256 `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`), upcast to F32:

```
llama-quantize --allow-requantize qwen2.5-0.5b-instruct-q4_k_m.gguf qwen05_f32.gguf F32
```

That gives 2,526,617,440 bytes, SHA256
`89ac62425c46039797e260ad4789adeda174320aa400e8b4b0d4a5e9aaa4ac25`. Master's own `llama-quantize`
reproduces that file byte for byte. The Q4 file stores the output matrix and the token embedding at
different quantization types, so after the upcast they are close but not identical
(`evidence/qwen2.5-0.5b-instruct/model_header.txt`). A conversion with a single tied tensor would train
289 tensors instead of 290, because llama skips the tensor named `token_embd.weight` when it marks
tensors to train (`src/models/qwen2.cpp:28-30`, `src/llama-context.cpp:3473`), so its numbers would not
match these digit for digit.

```
FT_NO_GRAD_RESET=1 FT_GRAD_PROBE=1 llama-finetune \
  -m qwen05_f32.gguf -f REPRO_window_synthetic.txt -o out.gguf \
  -c 512 -b 512 -ub 512 -ngl 0 -fa off -opt adamw -epochs 1 -lr 1e-30 -t 6
```

### Why `-lr 1e-30`

It is far too small to change a nonzero weight of these sizes in float32, so any growth in the gradient
comes from accumulation. A weight that is exactly zero could still move by about 1e-30, and nothing
printed here can show whether one did. The Results section shows what the logs do establish.

## Results

Two unrelated machines ran every build:

- **Linux:** Intel i7-10750H, AVX2, no AVX-512, Ubuntu 24.04, gcc 13.3.
- **Windows:** AMD Ryzen 9 9950X3D, AVX-512, Windows 11, MSVC 19.51.

**stories260K**, 462 steps:

| machine | arm | step 1 | step 2 | step 100 | step 462 | slope per step | R² |
|---|---|---|---|---|---|---|---|
| Linux | reset active | 0.400439 | 0.272711 | 0.249896 | 0.385401 | +6.1e-06 | 0.0001 |
| Linux | reset disabled | 0.400439 | 0.523719 | 21.4464 | 101.944 | +0.2209 | 0.9995 |
| Windows | reset active | 0.400439 | 0.272711 | 0.249896 | 0.385401 | +6.1e-06 | 0.0001 |
| Windows | reset disabled | 0.400439 | 0.52372 | 21.4464 | 101.944 | +0.2209 | 0.9995 |

**Qwen2.5-0.5B-Instruct**, 33 steps:

| machine | arm | step 1 | step 2 | step 33 | slope per step | R² |
|---|---|---|---|---|---|---|
| Linux | reset active | 2.77048 | 2.5036 | 1.84674 | +0.0199 | 0.0176 |
| Linux | reset disabled | 2.77048 | 5.27407 | 53.0931 | +1.6102 | 0.9974 |
| Windows | reset active | 2.77055 | 2.50368 | 1.84673 | +0.0199 | 0.0176 |
| Windows | reset disabled | 2.77055 | 5.27423 | 53.093 | +1.6102 | 0.9974 |

Slopes are ordinary least squares of `max|grad|` on step, and they agree between the machines to four
decimals, as do the R² values. No standard errors are given: the reset-disabled arm is a running sum,
so its residuals are not independent and a standard error would overstate the precision.

**The running sum shows in every step.** At every step *k*, on both models and both machines, the reset-disabled value is at most the sum of the reset-active values over steps 1 through *k*, with no exceptions. That is what a running sum must satisfy, since the maximum of a sum is at most the sum of the maxima. At the last step it is 101.9 against 143.7 on stories260K and 53.09 against 80.16 on Qwen, on both machines.

**On Qwen, step 2 adds up.** With the reset disabled, the largest gradient at step 2 is the largest at step 1 plus the largest at step 2 with the reset active. On Windows, 2.77055 + 2.50368 = 5.27423, the printed value exactly. On Linux, 2.77048 + 2.5036 = 5.27408 against a printed 5.27407, the same to within rounding in the last printed digit. The same tensor, `blk.3.ffn_up.weight`, holds the maximum in all three readings, which is consistent with a running sum. On stories260K the check does not apply, because different tensors hold the maximum at step 2 in the two arms (`blk.0.ffn_down.weight` and `output.weight`).

Step 1 is identical in both arms on every run, because nothing has accumulated yet. On Qwen it differs between the machines in the fifth significant figure (2.77048 on Linux, 2.77055 on Windows), as two independent floating-point paths can. On stories260K it is 0.400439 on both.

**The nonzero weights held still.** On stories260K every printed field of the train loss is identical between the two arms at all 462 steps, on both machines. On Qwen the running train loss agrees between the arms to five decimals at every step on both machines, ending at 2.22628; on Linux every field matches, and on Windows the `±` field differs in its last digit at steps 3 and 12. The largest weight is unchanged at every step of every run: 4.42509 on `output_norm.weight` for stories260K, and 214 on `blk.8.attn_k.bias` for Qwen. A later check on the Windows machine found one exception. On Qwen, 18,408,222 weights that are exactly zero in the F32 file moved in each arm, by at most 3.74e-29, in 121 of 291 tensors, and no nonzero weight moved (the per-tensor counts in `D_out_active_vs_input.txt` and `D_out_disabled_vs_input.txt` each sum to 18,408,222). With every exactly-zero weight set to 1e-20 first, nothing moved, the two arms printed the same train loss at every step, and the `±` difference at steps 3 and 12 went away. The logs and the two scripts that made them are in `evidence/zero_weights_windows/`.

## What is in here

| file | what |
|---|---|
| `PATCH_gradient-accumulator-reset.diff` | the fix alone, the form intended for upstream |
| `PATCH_reproduce.diff` | the fix with its off switch, and the probe, against `e6ab7c1a` |
| `PATCH_node_budget.diff` | the node budget, needed only for the Qwen reproduction |
| `PATCH_regression_test.diff` | `test_grad_dynamic` for `tests/test-opt.cpp`, against `e6ab7c1a` |
| `evidence/regression_test/` | per machine: `test-opt` output on master with the test (`master_plus_test.txt`) and with the test and the fix (`master_plus_test_plus_fix.txt`) |
| `REPRO_window_synthetic.txt` | the training data, described below |
| `evidence/stories260K/` | `model_header.txt`, and per machine: `stock_master.txt`, `reset_disabled.txt`, `reset_active.txt`, `binaries_stock.sha256`, `binaries_patched.sha256` |
| `evidence/qwen2.5-0.5b-instruct/` | `model_header.txt`, and per machine: `stock_master.txt`, `fix_and_probe_only.txt`, `reset_disabled.txt`, `reset_active.txt`, `meta.txt` (CPU, OS, compiler, model hash, data size), and the matching `binaries_*.sha256` |
| `evidence/zero_weights_windows/` | the Windows follow-up on Qwen: the two arms rerun with their output models kept (`D_keepout_active.txt`, `D_keepout_disabled.txt`), each output model compared with the input (`D_out_active_vs_input.txt`, `D_out_disabled_vs_input.txt`), the control with every exactly-zero weight set to 1e-20 (`E_make.txt`, `E_nozero_active.txt`, `E_nozero_disabled.txt`, `E_out_vs_input.txt`), and the two scripts (`gguf_tensor_diff.py`, `make_nozero_model.py`, whose first lines point at a local copy of llama.cpp's `gguf-py`; point them at yours) |

The run logs are as the programs wrote them. The runs were wrapped in coreutils `timeout`; on Linux the
logs of the aborted runs end with its report. In the Windows Qwen `meta.txt` the OS line was re-queried
after the run and the compiler line was reformatted from the same build's configure log; both are
marked.

**About the data.** 400 lines generated from a fixed four-slot template: 400 distinct lines, a
131-word vocabulary, and a closing sentence drawn from a pool of eight. 51,792 bytes, SHA256
`cd7753cd368a7e21fc603f4b6274e6d426b9cd3be3ef12cd177f35be06871f3d`. It is deliberately trivial and
contains no third-party text. It cannot support any claim about model quality, and none is made from
it.

## What is not claimed here

- No harm to a trained model is claimed. This report claims the accumulation, which the code path
  shows and the logs confirm. Showing damage needs a corpus that can carry it, and that measurement has
  not been made.
- Nothing is claimed about how #22704's fix behaves beyond what its diff shows.
- Nothing is claimed for models other than the two named here.
- The node-budget change is a reproduction aid, not a proposed fix.

## Earlier work on the same defect

Read from GitHub's API on 2026-09-22.

| PR | author | opened | state | commits | carries a fix for this defect |
|---|---|---|---|---|---|
| #22704 | srossitto79 | 2026-05-05 | open draft, merge conflicts, no reviews | 13 | yes, the earliest found |
| #22705 | srossitto79 | 2026-05-05 | open draft, merge conflicts, no reviews | 217 | yes, the same fix hunk as #22704 |
| #26794 | DFveloper | 2026-08-09 | closed without merge on 2026-08-11, with the comment "Way too many changes" | 215 | yes, the same fix hunk as #22704 |

srossitto79's is the earliest report we found. A code comment in #22704's diff gives the same
diagnosis: on the dynamic path the graph is rebuilt every call, the existing reset never runs, and the
accumulators carry over. That fix zeroes each accumulator directly, before the graph is rebuilt. The
fix here calls the existing reset on the graph after it is rebuilt. Both fire at the start of an
accumulation window.

In all three the fix rides inside a much larger change: a new op, a routed-matmul backward, and
optimizer work. None submits it alone, and none has been merged. #22705 and #26794 each carry more than
200 commits, 33 and 32 of them merges.

Separately, #27199 (ngxson, merged 2026-09-02) added `SET_ROWS` to the backward pass's in-place
assert. Its own description says `llama-finetune` stopped at that assert before it, and its example
trains stories260K, the model used above.

The fix here is offered on its own, in one hunk, with a regression test, and with reproductions built
from the published patches on two unrelated machines.

## License

MIT, matching llama.cpp.

---

*Drafted and fact-checked with an AI assistant from my own records, measurements and decisions. I read
every sentence and I am responsible for its contents. The patches were written with an AI
assistant at my direction and tested on my own machines.*
