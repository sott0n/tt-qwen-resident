// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Attention leader reader: per layer, waits for the streamers' projection row [q heads | gates | k | v]
// (1x32 tiles) and places head h of q and of the gate into row h of cb_qraw / cb_gate; then releases the
// group heads' partial sums to compute once all of them have written theirs.
//
// Runtime args: 0 row buffer address, 1 children (group heads of the reduction tree), 2 optional
// timeline buffer (0 = off; per layer 8 wall-clock words: [1] rows arrived, [2] rows placed,
// [3] partials arrived).

#include "api/dataflow/dataflow_api.h"
#include "attn_common.hpp"

using namespace resident_attn;

namespace {

// copy a 1x32 tile into row `row` of a 32x32 bf16 tile (two 16-element face rows)
FORCE_INLINE void place_row(uint32_t tiny, uint32_t tile, uint32_t row) {
    const uint32_t off = row_offset(row);
    noc_async_read(get_noc_addr(tiny), tile + off, kFaceRow);
    noc_async_read(get_noc_addr(tiny + kFaceRow), tile + kFace + off, kFaceRow);
}

FORCE_INLINE void zero_tiles(uint32_t cb, uint32_t n) {
    auto& iface = get_local_cb_interface(cb);
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(iface.fifo_limit - iface.fifo_size);
    for (uint32_t i = 0; i < n * kTile / 4; i++) {
        p[i] = 0;
    }
}

FORCE_INLINE void push_resident(uint32_t cb, uint32_t n) {
    cb_reserve_back(cb, n);
    cb_push_back(cb, n);
}

}  // namespace

void kernel_main() {
    const uint32_t rows = get_arg_val<uint32_t>(0);
    const uint32_t children = get_arg_val<uint32_t>(1);
    const uint32_t ts_addr = get_arg_val<uint32_t>(2);
    volatile tt_l1_ptr uint32_t* rows_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_rows));
    volatile tt_l1_ptr uint32_t* part_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_part));
    volatile tt_l1_ptr uint32_t* ts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ts_addr);
    volatile uint32_t* clk = reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    auto mark = [&](uint32_t l, uint32_t i) {
        if (ts_addr) {
            ts[l * 8 + i] = *clk;
        }
    };

    zero_tiles(cb_qraw, Dt);
    zero_tiles(cb_gate, Dt);
    push_resident(cb_mean, 1);
    push_resident(cb_qw, weight_sets * Dt);
    push_resident(cb_rope, 3);

    constexpr uint32_t gate0 = heads * Dt;
    for (uint32_t l = 0; l < layers; l++) {
        noc_semaphore_wait_min(rows_sem, num_streamers * (l + 1));
        mark(l, 1);
        cb_reserve_back(cb_qraw, Dt);
        cb_reserve_back(cb_gate, Dt);
        const uint32_t q = get_write_ptr(cb_qraw), g = get_write_ptr(cb_gate);
        for (uint32_t h = 0; h < heads; h++) {
            for (uint32_t d = 0; d < Dt; d++) {
                place_row(rows + (h * Dt + d) * kTiny, q + d * kTile, h);
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_qraw, Dt);
        for (uint32_t h = 0; h < heads; h++) {
            for (uint32_t d = 0; d < Dt; d++) {
                place_row(rows + (gate0 + h * Dt + d) * kTiny, g + d * kTile, h);
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_gate, Dt);
        mark(l, 2);

        noc_semaphore_wait_min(part_sem, children * (l + 1));
        mark(l, 3);
        push_resident(cb_parts, fanin * kPart);
    }
}
