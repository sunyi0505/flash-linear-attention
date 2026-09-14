# Ascend Triton traps

Silent-bug and compile/UB traps specific to Triton-Ascend. General FLA measurement traps stay in `fla-optimization-loop/references/TRAPS.md`. Case notes: [cases.md](cases.md). Last-dim leftover DMA padding: [last-dim-dma-pad.md](last-dim-dma-pad.md).

Each entry: **Fact / Why / How to apply**.

---

## Runtime DMA-path `if` keeps both sides in UB

**Fact:** A runtime `if is_tail_chunk:` that chooses `make_block_ptr` vs masked `tl.load` does **not** DCE the unused path on Triton-Ascend. Peak UB is the sum of both; Vector stays ~0.75 occupied; larger tiles fail compile even when MemoryUB bandwidth is free.

**Why:** The compiler treats the predicate as data-dependent, so both DMA sequences stay in the live set. Halo windows (`BT+W-1`) make the tail path even larger. The same class of bug: `if CONSTEXPR_FLAG or runtime:` around an optional pointer still compiles `ptr + …` when `ptr is None`.

**How to apply:** Host-split the last tile into a second launch with a `tl.constexpr` mode (never / always / runtime-for-varlen). Nest constexpr optional-pointer flags; do not OR them with runtime checks. See [cases.md § causal_conv1d](cases.md#causal_conv1dpy--1d-core-grid--constexpr-dma-split).

---

## Last-dim leftover DMA aliases the next row

**Fact:** CANN leftover DMA of a `block_ptr` tile is rounded to 32. If the last-dim stride is not a multiple of `max(tile, 32)`, that leftover writes into the **next row**. A partial last tile also needs `n + 32 <= padded` (D=60, tile=64 → pad **128**, not 64). Compile succeeds; only unaligned K/V/T or varlen is wrong. Related: masked / `boundary_check` stores RMW destination lanes (`empty` NaNs leak); leftover `g=0` then `exp2(0 - g_valid)` overflows to inf; leftover `tl.where` copies defeat `enable_ubuf_saving`.

**Why:** MTE leftover length is 32-aligned, not tile-aligned. DMA that starts at logical `n` still transfers 32 elements, so padding must cover `n + 32`, not just `ceil(n / tile) * tile`.

**How to apply:** `npu_pad` / `npu_unpad` / `npu_leftover_mask` from `fla.utils`. Kernel last-dim strides use `KS`/`VS`/`KP`/`VP`. `MASK_LEFTOVER` zeros padding lanes. Padded workspace is `zeros`, not `empty`. Mask gate diffs **before** `exp2`. Do not over-admit packed tiles on leftover. Full rule: [last-dim-dma-pad.md](last-dim-dma-pad.md).

---

## `constexpr` has no `.to()` — use `tl.cast` for address math

**Fact:** Specialized kernel args (`B`, `T`) and program IDs that fold (e.g. `i_t` when `NT==1`) are `constexpr`. `x.to(tl.int64)` is `AttributeError("'constexpr' object has no attribute 'to'")` at compile. CUDA kernels often write `i_t.to(tl.int64)` because those indices stay runtime there.

**Why:** Ascend specializes more integers than CUDA. `tl.cast(x, tl.int64)` works on constexpr and runtime ints; `.to()` only exists on tensor / load results.

**How to apply:** `t0 = tl.cast(i_t, tl.int64) * BT`, `bos = tl.cast(i_b, tl.int64) * T`, `tl.cast(B, tl.int64) * T`. Keep `tl.load(cu_seqlens + i_n).to(tl.int64)`. Never `(i_b * T).to(tl.int64)`.

---

## Int64 `make_block_ptr` offsets fail compile

**Fact:** `make_block_ptr` rejects int64 `offsets` / `block_shape` (`Block pointers only support 32 bit offsets/block_shape`). Feeding `t0 = tl.cast(i_t, tl.int64) * BT` as the row offset breaks compile after the overflow fix.

**Why:** Block-pointer metadata is int32 by design. Flattened `ptr + offset * stride` is the path that needs int64.

**How to apply:** Keep `t0` (int64) for `x + bos * D + t0 * D`. Pass `i_t * BT` (int32) to `make_block_ptr`. See [cases.md § causal_conv1d](cases.md#causal_conv1dpy--1d-core-grid--constexpr-dma-split).
