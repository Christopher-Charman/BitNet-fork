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
