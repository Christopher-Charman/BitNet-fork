# BitNet I2_S source qualification — 10 November 2026

## Scope and evidence

Source-only candidate for `Christopher-Charman/BitNet-fork` at `0b341e582afbf9e1011f24744b554c96a3477eb5`, with pinned `3rdparty/llama.cpp` at `390c307752ab78fd8189f359d6954c9ba1be74af`. No production model, service, or inference default is changed.

The original `quantize_i2_s` packs four adjacent values per byte. `dequantize_row_i2_s` and the AVX2 GEMV consumer expect a four-lane 128-value layout. An initial 128-element roundtrip produced **88 mismatches**.

The scalar dot fallback additionally used a signed ternary map despite the shared caller subtracting the activation sum from an encoded 0/1/2 dot. The source-only correction aligns quantization, dequantization, scalar and SIMD contracts.

## Reproduce

1. Check out the pinned BitNet-fork commit and its pinned llama.cpp submodule.
2. Run `python qualification/i2s/apply_pinned_llama_patch.py` from the repository root.
3. Configure and build `ggml-cpu` with CPU-only CMake settings. For scalar validation additionally set `GGML_NATIVE=OFF GGML_AVX=OFF GGML_AVX2=OFF GGML_AVX512=OFF` (CMake `-D` flags).
4. Set `BITNET_GGML_CPU_LIB` to the built `libggml-cpu.so` and run `python qualification/i2s/test_i2_s_roundtrip.py`.

The isolated Fasthosts qualification completed two builds: AVX2 and scalar. Both passed roundtrip and encoded GEMV parity for 128 and 256 values (four checks per build). Builds used a single low-priority CPU job and left the existing public service health endpoints returning HTTP 200.

Local qualified source commit: `6f7470433b156572f8e68513c44f06203131accf` in an isolated submodule worktree. The parent repository tracks the upstream submodule as a Git link, so this PR publishes a reproducible patch script and tests rather than incorrectly claiming that a submodule change has been integrated.

## Acceptance remaining

Independent semantic review of the patch and tests; exact-artifact readback; broader GEMM and model-level functional qualification; quality-preserving performance comparisons (useful throughput per CPU-second, RSS and latency). Merge or production activation is not authorized by this source-only publication.

## Published artifact readback — 10 November 2026

The published patch script was fetched from GitHub and applied to the original pinned llama.cpp source. The resulting `quants.c` matched the locally qualified corrected source **byte-for-byte** (`cmp`: PASS). The expanded published test suite passed 128-, 256-, and 512-element roundtrip and encoded GEMV parity cases on both the AVX2 and scalar CPU-only libraries (12 checks total). The pull request is ready for independent review; no reviewer approval or production release is implied.

## Extended SIMD group-boundary verification

The published regression was extended to 4096 elements (one complete 32-block SIMD group) and 4224 elements (a full group plus a 128-element tail). Both AVX2 and scalar builds passed packing roundtrip and encoded GEMV parity for all five lengths: 128, 256, 512, 4096 and 4224, **20 numerical checks total**. The source remains unmerged; the tests exercise numerical kernel correctness, not full-model quality or comparative performance.

## Multirow GEMM acceptance and scalar regression — 11 November 2026

The newly published `test_i2_s_gemm.py` exposed an additional functional defect in the scalar CPU fallback: multirow GEMM forwarded batched work to a scalar dot-product routine that did not implement the required output and activation strides. The first 4×4 test returned an incorrect value (-6 instead of 4). The isolated fix performs one encoded dot per output cell with explicit row/column addressing, preserving the SIMD implementation.

The corrected scalar build and the original AVX2 build both passed **nine GEMM configurations**: inner dimensions 128, 256 and 4096, crossed with 4×4, 5×3 and 3×5 row/column shapes (18 GEMM configurations across both builds). The published patch was fetched and applied to both pinned original C source files; each resulting file matched the qualified local source byte-for-byte. These checks supplement the prior 20 packing/GEMV checks. Independent review and model-level performance/quality qualification remain required before merge or production deployment.

## GEMV batch-boundary verification — 11 November 2026

The published `test_i2_s_gemv_rows.py` exercised 1, 4, 16, 17 and 32 weight rows at inner dimensions 128, 256 and 4096. All **30 configurations** (15 per AVX2/scalar build) passed encoded-dot parity, including both sides of the 16-row batching boundary. Across the published focused suites, the verified total is 68 configuration checks: 20 packing/single-row GEMV, 18 GEMM and 30 batched GEMV. This is kernel-level numerical evidence, not full-model or production acceptance.

## Bounded CPU kernel microbenchmark — 11 November 2026

The published `benchmark_i2_s_gemm.py` was retrieved from GitHub and run with `nice -n 19` against both locally built CPU backends. Each configuration used three warmups and 15 timed samples, with 8 weight rows and 8 activation columns. The measured **median kernel call latency** was:

| Inner dimension | AVX2 | Scalar |
| --- | ---: | ---: |
| 128 | 2.20 µs | 31.90 µs |
| 512 | 2.50 µs | 128.90 µs |
| 4096 | 4.40 µs | 1022.61 µs |

These are short, host-dependent microbenchmarks, not end-to-end inference throughput, CPU consumption, model quality or a release criterion. The scalar 4096-element samples were variable (minimum 621.80 µs); reproduce on a stable isolated host before making architecture-level performance decisions. No inference defaults were changed.

## Signed activation stress — 11 November 2026

The published `test_i2_s_signed_extremes.py` passed 10 additional GEMM configurations (five signed activation magnitudes × AVX2 and scalar). The magnitudes 3, 31, 63, 100 and 127 exercise the upper signed-int8 range at inner dimension 4096. All 4×4 output cells matched the integer encoded-dot reference. The cumulative focused configuration count is now **78** (68 prior plus 10 signed-int8 stress cases). No suitable full BitNet I2_S model artifact was located in the bounded local model search; model-level qualification is still outstanding.

## Maximum-positive accumulation — 11 November 2026

Published `test_i2_s_max_positive.py` was retrieved from the PR branch and passed on both AVX2 and scalar builds at inner dimensions 128 and 4096. Every 4×4 output cell matched encoded reference values 32,512 and 1,040,384 respectively. This adds four passing kernel configurations, for **82** focused configurations in total. The current PR has no submitted independent GitHub review. The established owned reviewer route must independently assess the exact PR head; numerical tests are not a substitute for semantic approval.

## Non-repacked matmul row-boundary defect — 11 November 2026

Source inspection identified an out-of-bounds final weight-row tile in `ggml-cpu.c`: the non-repacked I2_S fast path rounded `src0_end` upwards to a four-row boundary without clamping to `ne01`. For five physical rows, this could request eight rows. The pinned-source patch now applies `src0_end = MIN(src0_end, ne01)` before the empty-range check, preserving the physical row boundary. The published `test_i2_s_row_bounds.py` passed **528** deterministic row-partition configurations (1–33 rows × 1–16 threads), checking exact coverage and bounds. Both AVX2 and scalar `ggml-cpu` libraries rebuilt successfully; the 18 previously published GEMM parity configurations passed again. Fresh original pinned source was patched independently: `quants.c` and `ggml-cpu-i2s.c` matched the qualified sources byte-for-byte, while `ggml-cpu.c` differed only by the two explanatory comments accompanying the same clamp. This test verifies the row partition algorithm and source guard, not an end-to-end model execution.

## First real-model execution — 11 November 2026

Retrieved the official `microsoft/bitnet-b1.58-2B-4T-gguf` `ggml-model-i2_s.gguf` (1,187,801,280 bytes) into the isolated qualification build directory, SHA-256 `4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162`. Built `llama-cli` against the locally corrected CPU source with Miniconda and an explicit system `libgomp` link path. A first CLI run reached an interactive prompt and exceeded the bounded job duration; the corrected `--single-turn --simple-io` invocation exited successfully with 2 CPU threads, context 512 and 16 generated tokens. Observed reported rates were **30.6 prompt tokens/s** and **7.2 generated tokens/s** on this host. For the prompt `Write the word READY.`, the output was `Write the word READY / The word: 3 / 3 / I:` rather than a compliant answer. Thus **model load and generation are verified**, but **instruction-following quality is not qualified**. This is one short smoke sample, not a model capability verdict. No model or runtime default was changed; artifact is isolated in the qualification workspace.
