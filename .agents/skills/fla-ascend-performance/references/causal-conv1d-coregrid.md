# `causal_conv1d` Ascend 1D core-grid

Optimization recipe for the Triton-Ascend causal depthwise conv used by GDN / short convolution modules. Ported from the MindSpeed-ops pattern into FLA’s default backend (do not leave a separate MindSpeed dispatch as the only fast path).

Reference implementation: `fla/modules/backends/triton_ascend/causal_conv1d.py`
(`causal_conv1d_fwd/bwd_coregrid_kernel`, `_launch_*_coregrid`, `_can_use_coregrid`).

## Symptom (legacy path)

- Large packed shapes (e.g. T=8K, D=4K bf16) spend tens of ms in fwdbwd while a MindSpeed-style kernel finishes in ~1–2 ms.
- Profiler: many launches / small channel tiles, Vector + MTE dominated; host grid chunking under `ASCEND_MAX_GRID_DIM` when `B×NT×cdiv(D,BD)` is large.
- Separating silu / residual / bias as host kernels adds launch + writeback traffic on the hot path.

Measured (default `fla`, bf16, synced fwdbwd; approximate):

| Shape | Legacy | Core-grid |
|-------|--------|-----------|
| T=2048, D=1024 | ~3.3 ms | ~0.5–0.7 ms |
| T=8192, D=4096 | ~52 ms | ~1.6 ms |

## Root causes addressed

1. **Multi-axis grid + tiny BD** → launch / grid-product overhead and poor Vector occupancy.
2. **Weight apply via full broadcast** → extra UB pressure vs tap-wise `extract_slice`.
3. **Unfused bias / silu / residual** → intermediate MTE writebacks.
4. **Weight layout mismatch** → FLA public `[D, W]` is not the contiguous `[W, D]` the fast kernels expect.

## Recipe

### 1. 1D core-grid on **vector cores**

This op is Vector-bound (depthwise tap MAC + silu), not Cube. Launch:

```python
NUM_CORES = driver...get_device_properties(device)['num_vectorcore']
kernel[(NUM_CORES,)](...)
```

Inside the kernel, flatten work and stride by program count:

```text
total_tasks = NUM_BLKS_D * NUM_CHKS
for task_id in range(pid, total_tasks, num_programs):
    i_d = task_id % NUM_BLKS_D
    i_chk = task_id // NUM_BLKS_D
    # decode i_chk → (i_b/i_n, i_t) or varlen via chunk_indices
```

Do **not** copy the Cube `num_aicore` recipe blindly for this op. One launch, no host grid-chunk loop, better balance when `NUM_CHKS` is irregular (varlen).

### 2. Channel tile must **exactly divide D**

```python
def _select_coregrid_bd(D, preferred):
    bd = min(preferred, next_power_of_2(max(D, 1)))
    while bd > 8 and (bd > D or D % bd != 0):
        bd //= 2
    if bd > D or D % bd != 0:
        return None  # fall back to legacy
    return bd
```

- Fwd preferred BD: **256**.
- Bwd preferred BD: **64** when `cdiv(D,64)*NUM_CHKS > NUM_CORES//2`, else **32** (enough tasks to fill waves).
- BT: `min(32, next_power_of_2(cdiv(max(16, B*T), NUM_CORES)))`.
- Odd widths (e.g. D=200) **must** stay on the legacy path — masking alone is not enough; bwd `dw` numerics break when BD does not land on exact D boundaries.

Gate helper: `_can_use_coregrid` also requires a usable BD (`>= 16` at preferred 64).

### 3. Weight layout + `extract_slice` / `insert_slice`

- Public FLA weight is `[D, W]`. Host-transpose to `[W, D].contiguous()` before launch; after bwd, `dw.sum(0).to(weight).transpose(0, 1)`.
- Load `b_w` once per task via `tl.make_block_ptr(weight, (W, D), (D, 1), ...)`.
- Apply each tap with Ascend extension ops (shim onto `tl` if needed):

```python
from triton.language.extra.cann.extension import extract_slice, insert_slice
b_yi *= tl.extract_slice(b_w, [i_w + W - 1, 0], [1, BD], [1, 1])
```

Bwd accumulates `b_dw` with `insert_slice` / per-tap stores. Avoid materializing a full `[W, BT, BD]` weight broadcast in UB.

### 4. Fuse bias / activation / residual on fwd; unify bwd

- Fwd: fp32 accum → optional bias → silu/swish → residual → RTNE cast store.
- Bwd hot path (no state): one kernel for `dx` / `dw` / `db`; silu grad uses a precomputed linear `y` (`_launch_fwd_coregrid(..., activation=None)`).
- Keep `dr = dy` when residual is present (same as legacy semantics).

### 5. Hot-path eligibility (keep narrow)

Use core-grid only when **all** hold:

| Condition | Why |
|-----------|-----|
| `x` (and `residual` if any) contiguous, `stride(-1)==1` | Packed BTD loads / block_ptr |
| `initial_state is None` and `dht is None` | State branches need smaller UB / different validation |
| `_select_coregrid_bd(D, …)` succeeds | Exact channel tiling |
| not `layout_fallback` | Caller forced legacy |

Otherwise keep the existing legacy kernels (update / cache / decode / odd D / state).

When the gate passes, **force `dy.contiguous()`** before bwd launch. Mixing contiguous vs strided `dy` across coregrid vs legacy causes bf16 gradient mismatches (e.g. cat/split views in GDN).

### 6. Tail-of-allocation / MTE safety

Packed storage is always `[B, T, D]` with `TOTAL_ROWS = B*T`. For the last chunk, `block_ptr` extents can address past the allocation → MTE “DDR address out of range”.

```text
is_tail_chunk = (bos + i_t*BT + BT) > B*T          # fwd
is_tail_chunk = (bos + i_t*BT + BT + W - 1) > B*T  # bwd (wider dy window)
```

On the tail path: masked pointer loads/stores instead of `make_block_ptr`. Non-tail: `make_block_ptr` + `boundary_check`.

### 7. UB hygiene inside the tap loop

- Load each `x` tap **inside** `tl.static_range` — preloading all W taps overflows UB under large `BT×BD`.
- Bwd `dh0` (legacy/state) should load `dy` row-by-row for the same reason.
- No `num_warps` / `num_stages` on Ascend launches.

## Correctness gate

```bash
# Frozen module gate for this op
pytest tests/modules/test_conv.py -k "not cuda" -q
```

Also spot-check: non-aligned D (legacy), varlen, residual+silu, non-contiguous `dy`, and a large training shape (T≥2048, D≥1024) timing after changes.

## Anti-patterns

- Enabling core-grid for every D via masks only (breaks `dw` when `D % BD != 0`).
- Leaving MindSpeed as a separate env-selected impl while FLA default stays slow.
- Skipping `dy.contiguous()` and falling back to legacy for strided grads only.
- Using `num_aicore` for this Vector op, or copying CUDA `num_warps`/`num_stages`.
- Preloading all conv taps into UB “for MTE efficiency”.
- Expanding the hot path to `initial_state`/`dht` without re-validating UB + full `test_conv` (kernel has branches; host gate intentionally disables them).

## Follow-ups (optional)

- Revisit state / `dht` / update kernels with a reduced BD budget if profiling shows they matter.
- Calibrate fwd BD 256 vs 128 on mid-size D when task count is already ≫ cores (wave-count vs per-task cost).
- Keep generalizable lessons (exact-divisor BD, vectorcore grid, extract_slice, tail DMA) in this skill; do not bundle doc-only edits into the perf PR.
