# Optimizing Mixed SIMD and SIMT Operators

## Scope

When applying these techniques to FLA Triton Ascend, first verify compiler support and inspect the generated code. Ascend C address-space qualifiers and data transfer APIs cannot be used directly in Triton; loads and conditional expressions in the source code do not establish the final storage location or instruction selection.

## 1. Establish a Baseline and Choose an Approach

1. Record the device, CANN version, compiler, target function, shape, dtype, layout, core count, thread configuration, workload per core, and peak UB usage.
2. Define the input distribution, index range, and exceptional-value semantics. Establish a baseline using the main skill's collector and existing tests.
3. If SIMT performs many scattered GM reads from a small input, evaluate staging the input in UB before accessing it.
4. If SIMD expresses elementwise conditions through long sequences of `Compare`, `Select`, and `MaskAnd`, evaluate rewriting them as SIMT branches.
5. Change only one major factor per iteration and compare total kernel execution time. Evaluate combinations only after each change proves effective independently.

Do not add UB staging automatically when accesses are contiguous and cache reuse is effective. If other computation is the bottleneck, address that bottleneck first.

## 2. Optimize Random Memory Access

Prerequisites: the data accessed randomly must be much smaller than the actual available UB capacity, and the access savings must outweigh the added transfer and synchronization costs.

1. Calculate the number of bytes to cache, accounting for space used by other inputs, outputs, temporary data, alignment, and double buffering.
2. Transfer the candidate data from GM to UB using a SIMD transfer interface supported by the target version.
3. Allow SIMT to read UB only after synchronization between the transfer and computation is complete.
4. Use the correct UB address space for the corresponding SIMT input. Decide whether to stage indices and outputs based on measurements.
5. Retain thread-boundary and index upper-bound checks to ensure `index < INPUT_SIZE`.
6. Compare total execution time, including transfers, synchronization, and writeback, and use the pipeline timeline to explain the changes.

For the platform described in the official documentation, SIMT accesses GM at a granularity of 128 B, and the available UB upper limit is estimated as `256 KB - 8 KB - 32 KB = 216 KB`. These values are not constants across platforms, and the actual budget must also account for the operator's own allocations. FLA implementations should follow the device capacity management approach in the main skill.

Relevant official Ascend C interfaces include `asc_copy_gm2ub_align` for data transfer, `asc_sync_notify` / `asc_sync_wait` for synchronization from `PIPE_MTE2` to `PIPE_V`, and `asc_vf_call` for invoking SIMT functions. Verify signatures, lengths, alignment, and event requirements before use; do not infer arguments from interface names.

If the entire dataset does not fit in UB, tiling is a candidate approach that requires validation. First check whether random indices cross tile boundaries and assess the cost of repeated transfers.

## 3. Optimize Branch Computation

1. Recover the full mathematical and exceptional-value semantics of the existing SIMD comparisons, masks, and selections.
2. Rewrite portions suited to elementwise decisions as SIMT conditional logic, while retaining an appropriate implementation for regular bulk computation.
3. Keep the core count, workload per core, data placement, and input distribution fixed when comparing total kernel execution time.
4. Test different branch distributions; shorter code does not establish a performance improvement.

Integer Floor Mod: first compute a valid integer remainder `rem = a % b`. If `rem != 0` and the remainder and divisor have opposite signs, return `rem + b`; otherwise, return `rem`. Examples: `(-7, 3) -> 2`, `(7, -3) -> -2`, and `(6, -3) -> 0`. This rule cannot directly replace floating-point Floor Mod.

First define how to handle a zero divisor and division of the minimum negative integer by `-1`. The official SIMD snippet includes selection logic that returns `-1` for a zero divisor, while the SIMT snippet does not explicitly handle zero divisors. They therefore cannot be assumed equivalent for all inputs.

## 4. Validate and Report

- Cover empty inputs, tail tiles, the last valid element, invalid indices equal to the upper bound, combinations of positive and negative operands, zero remainders, and integer extremes.
- Check UB capacity, lifetimes, and synchronization between transfers and reads. Follow the main skill to validate supported gradients, optional parameters, and variable-length inputs, preserving existing tolerances.
- Compare total kernel execution time using the same environment, warmup, synchronization, and inputs. Repeat microsecond-scale measurements to identify noise.
- Explain gains using `SIMT_LDG`, MTE2, and memory load metrics. Lower bandwidth on the Dcache path may indicate that traffic has shifted to UB; interpret it together with total execution time.
- Report bottleneck evidence, how the change works, platform constraints, correctness results, execution times before and after the change, and the scope of applicability. Compute the execution-time reduction as `(baseline - optimized) / baseline`, and distinguish it from the speedup factor.
- Do not claim success when results regress or are unstable. Adjust or revert only your own candidate changes. If testing on the device is unavailable, explicitly mark the results as unmeasured.

## Official Examples and Code Caveats

- Gather: input of `8192 float` (32 KiB), with 65536 elements each for indices and output, using 64 thread blocks with 1024 threads per block. Caching the input reduces execution time from approximately `4.56 us -> 3.57 us`, a reduction of about 21.71%. L2-to-Dcache bandwidth changes from `10.04 -> 1.61 GB/s`, while L2-to-UB bandwidth after optimization is `12.93 GB/s`.
- Floor Mod: with the same core count, workload per core, and data placement in UB, SIMD takes `4.03 us` and SIMT takes `3.444 us`, an execution-time reduction of about 14.6%. The corresponding section does not specify the full test shape.
- The incorrect Gather example uses `gather_idx > INPUT_SIZE`, which allows an index equal to the upper bound. The correct early-return condition is `gather_idx >= INPUT_SIZE`.
- The Floor Mod SIMT snippet uses only `threadIdx.x` and does not check bounds using the length parameter. A complete implementation must add indexing and guards appropriate to the actual thread organization.

These performance figures come from the official examples and have not been remeasured in the current environment. They do not guarantee gains for other operators.

## Official Sources

- [Optimizing Mixed SIMD and SIMT Operators](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_best_practices_10_10026.html)
- [Improving Memory Access Efficiency with Unified Buffer](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_best_practices_10_10029.html)
- [Implementing Conditional Branching with SIMT](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_best_practices_10_10030.html)
- [Complete Gather Example](https://gitcode.com/cann/asc-devkit/tree/9.0.0/examples/01_simd_cpp_api/04_best_practices/03_fusion_compute_practices/simt_gather_with_ub)
- [Complete Floor Mod Example](https://gitcode.com/cann/asc-devkit/tree/9.0.0/examples/01_simd_cpp_api/04_best_practices/03_fusion_compute_practices/simt_and_simd_floor_mod)
