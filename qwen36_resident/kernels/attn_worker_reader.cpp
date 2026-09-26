// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Attention worker reader. Per layer (set = layer % weight_sets):
//   chunk worker: reads this core's chunk of the set's KV cache (complete position tiles only; see
//     attn_common.hpp for the layout);
//   tail core: prefetches the set's tail k | v row (bank, row j), waits for the streamers'
//     projection row and places k and v into row 0 of cb_kraw / cb_vraw;
// then releases the leader's multicast q and global row max, and a group head's children's partials,
// to compute.
//
// Runtime args: 0 role (0 idle, 1 chunk worker, 2 tail core), 1 n_tiles, 2 bank, 3 j, 4 J (this core
// reads rows j, j + J, ... of the bank's shard), then weight_sets x (k cache address, v cache
// address), then children, then an optional
// timeline buffer (0 = off; per layer 4 wall-clock words: [0] KV read, [1] q arrived, [2] M arrived),
// then (tail core) the row buffer address, r (row of this position in the tail tile).

#include "api/dataflow/dataflow_api.h"
#include "attn_common.hpp"

using namespace resident_attn;

namespace {

FORCE_INLINE uint32_t cb_base(uint32_t cb) {
    auto& iface = get_local_cb_interface(cb);
    return iface.fifo_limit - iface.fifo_size;
}

FORCE_INLINE void push_resident(uint32_t cb, uint32_t n) {
    cb_reserve_back(cb, n);
    cb_push_back(cb, n);
}

// copy a 1x32 tile into row 0 of a 32x32 bf16 tile (two 16-element face rows)
FORCE_INLINE void place_row0(uint32_t tiny, uint32_t tile) {
    noc_async_read(get_noc_addr(tiny), tile, kFaceRow);
    noc_async_read(get_noc_addr(tiny + kFaceRow), tile + kFace, kFaceRow);
}

// a bf16 32x32 tile with `on` where one(row, col) holds and `off` elsewhere
FORCE_INLINE void fill_bf16(uint32_t addr, bool (*one)(uint32_t, uint32_t), uint16_t on, uint16_t off) {
    volatile tt_l1_ptr uint16_t* m = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(addr);
    for (uint32_t row = 0; row < 32; row++) {
        for (uint32_t col = 0; col < 32; col++) {
            m[((row / 16) * 2 + col / 16) * 256 + (row % 16) * 16 + col % 16] = one(row, col) ? on : off;
        }
    }
}

uint32_t tail_r;

}  // namespace

void kernel_main() {
    const uint32_t role = get_arg_val<uint32_t>(0);
    if (role == 0) {
        return;
    }
    const uint32_t n_tiles = get_arg_val<uint32_t>(1);
    const uint32_t bank = get_arg_val<uint32_t>(2);
    const uint32_t j = get_arg_val<uint32_t>(3);
    const uint32_t J = get_arg_val<uint32_t>(4);
    constexpr uint32_t cache_base = 5;
    const uint32_t children = get_arg_val<uint32_t>(cache_base + 2 * weight_sets);
    const uint32_t ts_addr = get_arg_val<uint32_t>(cache_base + 2 * weight_sets + 1);
    constexpr uint32_t tail_args = cache_base + 2 * weight_sets + 2;
    volatile tt_l1_ptr uint32_t* q_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_q));
    volatile tt_l1_ptr uint32_t* M_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_M));
    volatile tt_l1_ptr uint32_t* part_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_part));
    volatile tt_l1_ptr uint32_t* rows_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_rows));
    volatile tt_l1_ptr uint32_t* ts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ts_addr);
    volatile uint32_t* clk = reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    auto mark = [&](uint32_t l, uint32_t i) {
        if (ts_addr) {
            ts[l * 4 + i] = *clk;
        }
    };
    push_resident(cb_one, 1);

    uint32_t rows = 0;
    if (role == 2) {
        rows = get_arg_val<uint32_t>(tail_args);
        tail_r = get_arg_val<uint32_t>(tail_args + 1);
        for (uint32_t cb : {cb_kraw, cb_vraw}) {
            volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_base(cb));
            for (uint32_t i = 0; i < Dt * kTile / 4; i++) {
                p[i] = 0;
            }
        }
        // causal mask over the tail columns (0 up to r, -inf after); R = ones in row r, 1 - R
        cb_reserve_back(cb_mask, 1);
        fill_bf16(get_write_ptr(cb_mask), [](uint32_t, uint32_t col) { return col <= tail_r; }, 0, 0xff80);
        cb_push_back(cb_mask, 1);
        cb_reserve_back(cb_rowsel, 2);
        fill_bf16(get_write_ptr(cb_rowsel), [](uint32_t row, uint32_t) { return row == tail_r; }, 0x3f80, 0);
        fill_bf16(get_write_ptr(cb_rowsel) + kTile, [](uint32_t row, uint32_t) { return row == tail_r; }, 0, 0x3f80);
        cb_push_back(cb_rowsel, 2);
        push_resident(cb_mean, 1);
        push_resident(cb_kw, weight_sets * Dt);
        push_resident(cb_rope, 3);
    }

    // rows first, first + step, ... of the bank's shard, one kv_row each
    auto read_rows = [&](uint32_t bank_id, uint32_t addr, uint32_t first, uint32_t step, uint32_t n, uint32_t l1) {
        for (uint32_t k = 0; k < n; k++, l1 += kKvRow) {
            noc_async_read(get_noc_addr_from_bank_id<true>(bank_id, addr + (first + k * step) * kKvRow), l1, kKvRow);
        }
    };

    constexpr uint32_t k0 = 2 * heads * Dt, v0 = k0 + Dt;
    for (uint32_t l = 0; l < layers; l++) {
        const uint32_t set = l % weight_sets;
        const uint32_t k_addr = get_arg_val<uint32_t>(cache_base + 2 * set);
        const uint32_t v_addr = get_arg_val<uint32_t>(cache_base + 2 * set + 1);
        if (role == 1) {
            cb_reserve_back(cb_k, chunk_max * Dt);
            cb_reserve_back(cb_v, chunk_max * Dt);
            read_rows(bank, k_addr, j, J, n_tiles, get_write_ptr(cb_k));
            read_rows(bank, v_addr, j, J, n_tiles, get_write_ptr(cb_v));
            noc_async_read_barrier();
            cb_push_back(cb_k, chunk_max * Dt);
            cb_push_back(cb_v, chunk_max * Dt);
        } else {
            cb_reserve_back(cb_tail8, 2 * Dt);
            read_rows(bank, k_addr, j, 1, 1, get_write_ptr(cb_tail8));
            read_rows(bank, v_addr, j, 1, 1, get_write_ptr(cb_tail8) + kKvRow);
            noc_async_read_barrier();
            cb_push_back(cb_tail8, 2 * Dt);
            noc_semaphore_wait_min(rows_sem, num_streamers * (l + 1));
            cb_reserve_back(cb_kraw, Dt);
            cb_reserve_back(cb_vraw, Dt);
            const uint32_t k = get_write_ptr(cb_kraw), v = get_write_ptr(cb_vraw);
            for (uint32_t d = 0; d < Dt; d++) {
                place_row0(rows + (k0 + d) * kTiny, k + d * kTile);
                place_row0(rows + (v0 + d) * kTiny, v + d * kTile);
            }
            noc_async_read_barrier();
            cb_push_back(cb_kraw, Dt);
            cb_push_back(cb_vraw, Dt);
        }
        mark(l, 0);
        noc_semaphore_wait_min(q_sem, l + 1);
        mark(l, 1);
        push_resident(cb_q, Dt);
        noc_semaphore_wait_min(M_sem, l + 1);
        mark(l, 2);
        push_resident(cb_M, 1);
        if (children > 0) {
            noc_semaphore_wait_min(part_sem, children * (l + 1));
            push_resident(cb_parts, fanin * kPart);
        }
    }
}
