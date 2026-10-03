// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Prefill -> decode conv-history handoff (BRISC, one core per GDN copy range, every chip). The host packs
// this chip's history columns compactly: 64 B page ((g * 3 + slot) * K + k) holds the 32 columns of conv
// tile k for GDN copy g and ring slot `slot`. The decode's history tensor (row-major, one page per
// (streamer, copy) row of 3 slots x 32 1x32 tiles) takes it at row rows[k] * copies + g, element
// slot * 1024 + tiles[k] * 32.
//
// Compile-time args: 0 K, 1 copies. Runtime args: 0 compact address, 1 history address, 2 first copy,
// 3 copies of this core, then K x (row, tile).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t K = get_compile_time_arg_val(0);
    constexpr uint32_t copies = get_compile_time_arg_val(1);
    constexpr uint32_t chunk = 64, row_bytes = 3 * 1024 * 2, cb_buf = 0;
    const uint32_t compact = get_arg_val<uint32_t>(0), hist = get_arg_val<uint32_t>(1);
    const uint32_t first = get_arg_val<uint32_t>(2), count = get_arg_val<uint32_t>(3);
    const InterleavedAddrGen<true> src{.bank_base_address = compact, .page_size = chunk};
    const InterleavedAddrGen<true> dst{.bank_base_address = hist, .page_size = row_bytes};
    const uint32_t buf = get_write_ptr(cb_buf);

    for (uint32_t g = first; g < first + count; g++) {
        for (uint32_t slot = 0; slot < 3; slot++) {
            for (uint32_t k = 0; k < K; k++) {
                noc_async_read(get_noc_addr((g * 3 + slot) * K + k, src), buf + k * chunk, chunk);
            }
            noc_async_read_barrier();
            for (uint32_t k = 0; k < K; k++) {
                const uint32_t row = get_arg_val<uint32_t>(4 + 2 * k), tile = get_arg_val<uint32_t>(5 + 2 * k);
                noc_async_write(
                    buf + k * chunk, get_noc_addr(row * copies + g, dst, (slot * 1024 + tile * 32) * 2), chunk);
            }
            noc_async_write_barrier();
        }
    }
}
