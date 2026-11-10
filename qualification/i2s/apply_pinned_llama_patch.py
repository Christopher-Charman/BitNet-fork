#!/usr/bin/env python3
"""Apply the isolated I2_S source correction to the pinned llama.cpp submodule.

This script modifies only 3rdparty/llama.cpp/ggml/src/ggml-cpu/quants.c.
Run from the BitNet-fork repository root. Requires the pinned submodule checkout.
"""
from pathlib import Path

path = Path("3rdparty/llama.cpp/ggml/src/ggml-cpu/quants.c")
source = path.read_text()
replacements = [
    (
        """        int byte_idx = i / 4;
        int bit_pos = 6 - 2 * (i % 4);
        q[byte_idx] |= (val << bit_pos);""",
        """        // I2_S packs 128 values as four 32-element SIMD lanes.
        const int64_t block = i / 128;
        const int64_t local = i % 128;
        const int64_t byte_idx = block * 32 + local % 32;
        const int bit_pos = 6 - 2 * (int)(local / 32);
        q[byte_idx] |= (val << bit_pos);""",
    ),
    (
        """            int byte_idx = i / 4;
            int bit_pos = 6 - 2 * (i % 4);
            int w = map2bit[(x[row * (n/4) + byte_idx] >> bit_pos) & 0x03];""",
        """            const int block = i / 128;
            const int local = i % 128;
            const int byte_idx = block * 32 + local % 32;
            const int bit_pos = 6 - 2 * (local / 32);
            int w = map2bit[(x[row * (n/4) + byte_idx] >> bit_pos) & 0x03];""",
    ),
    (
        """    static const int map2bit[4] = { -1, 0, 1, 0 };""",
        """    // The shared caller subtracts activation sum from encoded dot.
    static const int map2bit[4] = { 0, 1, 2, 1 };""",
    ),
]
for before, after in replacements:
    count = source.count(before)
    if count != 1:
        raise SystemExit(f"Expected exactly one matching source block, found {count}: {before[:60]}")
    source = source.replace(before, after)
path.write_text(source)
print(f"Updated {path}")
