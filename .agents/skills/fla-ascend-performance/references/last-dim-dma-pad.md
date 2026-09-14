# Last-dim leftover DMA padding (`NPU_ALIGN=32`)

Apply when a Triton-Ascend kernel uses `make_block_ptr` (or a last-dim stride equal to logical `K`/`V`) and `K`, `V`, or `T` is not a multiple of the tile — including varlen. Wrong last-dim padding is a **silent accuracy bug**: compile succeeds, aligned shapes pass, unaligned / leftover tiles do not.

Helpers (do not re-implement): `npu_pad`, `npu_unpad`, `npu_leftover_mask` in `fla/utils/_ascend_align.py`, re-exported from `fla.utils`.

Reference implementations: `chunk_delta_h.py`, `chunk_h.py`, `chunk_o.py`, GDN/KDA `wy_fast.py`, GLA `chunk.py`, KDA `chunk_bwd.py` / `fused_recurrent.py`.

## Symptom

- Numeric mismatch vs Torch only when `K`/`V`/`T` is not tile-aligned (e.g. D=60/100, `T % BT != 0`) or on varlen; power-of-two D=128 looks fine.
- NaN poisoning of `empty` workspace leaks into valid leftover-chunk tokens.
- Gated kernels: inf in the causal tile from `exp2(0 - g_valid)` when leftover `g` loads as 0.
- Leftover `tl.where` then UB-overflows a tile that compiled on the aligned path (`enable_ubuf_saving` packing drops).

## Root cause

CANN leftover DMA of a tile is **rounded to 32**. If the last-dim stride is not a multiple of `max(tile, 32)`, that leftover aliases the **next row**.

A partial last tile is worse: leftover DMA starts at logical `n` and is 32 wide, so the row must also satisfy `n + 32 <= padded` (e.g. D=60, tile=64 needs **128**, not 64).

Triton-Ascend masked / `boundary_check` stores can **RMW destination lanes**. Uninitialized `empty` buffers and a last-chunk leftover that spills into the next batch both become visible outputs.

## Alignment rule

```python
# npu_pad(n, tile): ceil to align = max(tile, 32), then +align if n % 32 != 0 and n + 32 > padded
KS, VS = npu_pad(K, BK), npu_pad(V, BV)   # state last dims
KP, VP = npu_pad(K, BK), npu_pad(V, BV)   # q/k/v/o / grad last dims
```

| Logical `n` | Tile | `npu_pad(n, tile)` | Why |
|-------------|------|--------------------|-----|
| 128 | 128 | 128 | already multiple of `max(tile, 32)` |
| 64 | 64 | 64 | same |
| 60 | 64 | **128** | leftover DMA at 60 is 32 wide; 60+32=92 > 64 |
| 100 | 128 | **256** | ceil to 128, then 100+32=132 > 128 so add another 128 |

`npu_pad(100, 128)`: `align=128`, `padded=128`, `100 % 32 != 0` and `132 > 128` → **256**.

Kernel pointer math and `make_block_ptr` **shape/stride** use `KS`/`VS`/`KP`/`VP`. Logical `K`/`V` stay in masks (`o_k < K`) and in `npu_unpad` after the store.

## How to apply

### 1. Host pad workspace / state, unpad on return

```python
mask_leftover = npu_leftover_mask(T=T, BT=BT, K=K, BK=BK, V=V, BV=BV, varlen=cu_seqlens is not None)
KS, VS = npu_pad(K, BK), npu_pad(V, BV)
# pad incoming state last dims; allocate h / ht / o / dq / … with padded last dim
# return npu_unpad(o, V), npu_unpad(h, K, V)  # or (V, K) when STATE_V_FIRST
```

`MASK_LEFTOVER` is a `tl.constexpr`. Aligned K/V/T with no varlen skip leftover-mask copies and keep large Cube tiles (BK/BV=128).

### 2. Zero leftover lanes when `MASK_LEFTOVER`

After loads that can include padding / OOB: `tl.where(mask, tile, 0)`. Stores of padded last-dim tensors use masked `tl.store` or `block_ptr` over the **padded** extent so leftover DMA lands in padding, not the next row.

### 3. `zeros`, not `empty`, when leftover or pad ≠ logical

```python
# Triton-Ascend masked/boundary stores can RMW destination lanes.
o = v.new_zeros(B, T, HV, VS) if mask_leftover or VS != V else v.new_empty(B, T, HV, VS)
```

NaN-poisoned `empty` leaks into valid leftover-chunk tokens. Leftover-T of one batch’s last chunk can RMW the next batch.

### 4. Mask gate diffs **before** `exp2`

Leftover chunks load OOB `g` as 0. Large `|g|` (e.g. `gln=0.01`) makes `exp2(0 - g_valid)` overflow to inf and contaminate the causal tile.

```python
m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
b_g_diff = tl.where(m_A, b_g[:, None] - b_g[None, :], 0)
b_A = b_A * exp2(b_g_diff)
```

Do not compute `exp2(b_g[:, None] - b_g[None, :])` first and mask after.

### 5. Leftover UB: do not over-admit tiles that only compile packed

Leftover `tl.where` copies (and in-kernel `b_h *= gk`) defeat `enable_ubuf_saving`. D256 `BK=256/BV=128` compiles on the aligned packed path (~256KiB live set packed into 192KiB) and fails on leftover. Feed leftover / `use_gk` into the host peak model (`_FWD_UB_LEFTOVER=1.00`, extra live-slab copies) and pick smaller tiles on that path only.

## Anti-patterns

| Attempt | Result |
|---------|--------|
| Last-dim stride = logical `K`/`V` when `n % 32 != 0` | Leftover DMA aliases the next row |
| `npu_pad` = `cdiv(n, tile)*tile` only | D=60/tile=64 pads to 64; leftover 60:92 still overruns |
| `torch.empty` for leftover / padded outputs | RMW of NaN / next-batch tokens |
| `exp2(g_i - g_j)` then causal mask | inf from padding `g=0` |
| Same BK/BV on leftover as on aligned packed path | compile UB overflow (`tl.where` kills packing) |
| Keep leftover `tl.where` on the aligned hot path | extra copies, worse UB / tiles |
| Return padded tensors to the public API | shape mismatch vs CUDA / Torch |

## When to apply elsewhere

Any new `triton_ascend` kernel that:

1. Stores through `make_block_ptr` along a last dim of size `K` or `V`, or
2. Holds recurrent state with last dims `(K, V)` / `(V, K)`, or
3. Uses `exp2` of gate differences on a `BT×BT` tile.

Correctness gate: unaligned K/V/T **and** varlen, under NaN poisoning — not only D=64/128 dense.
