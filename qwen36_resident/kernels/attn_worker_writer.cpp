// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Attention worker writer: per layer, sends this core's row max (rows 0..heads-1 of column 0) to the
// leader, then its partial (a group head: its partial plus its children's) to slot `slot` of its
// parent's cb_parts (the leader or a group head), top two faces of each tile only. A parent's cb_parts
// address arrives in this core's sem_addr; a group head sends its own to its children at start. The
// tail core then writes the updated tail k | v tiles back to the set's DRAM cache (off the critical path).
//
// Runtime args: 0 role (0 idle, 1 chunk worker, 2 tail core), 1 row-max slot index w, 2 leader x,
// 3 leader y (NOC coordinates), 4 leader row-max slots address, 5 parent x, 6 parent y, 7 slot at the
// parent, 8 children, then children x (x, y), then an optional timeline buffer (0 = off; word [3] of the
// reader's: partial sent), then (tail core) the tail row's bank and row, and weight_sets x (k cache,
// v cache).

#include "api/dataflow/dataflow_api.h"
#include "attn_common.hpp"

using namespace resident_attn;

void kernel_main() {
    const uint32_t role = get_arg_val<uint32_t>(0);
    if (role == 0) {
        return;
    }
    const uint32_t w = get_arg_val<uint32_t>(1);
    const uint32_t lx = get_arg_val<uint32_t>(2);
    const uint32_t ly = get_arg_val<uint32_t>(3);
    const uint64_t m_dst = get_noc_addr(lx, ly, get_arg_val<uint32_t>(4) + w * kMSlot);
    const uint32_t px = get_arg_val<uint32_t>(5);
    const uint32_t py = get_arg_val<uint32_t>(6);
    const uint32_t slot = get_arg_val<uint32_t>(7);
    const uint32_t children = get_arg_val<uint32_t>(8);
    const uint32_t ts_addr = get_arg_val<uint32_t>(9 + 2 * children);
    const uint32_t tail_args = 10 + 2 * children;
    volatile tt_l1_ptr uint32_t* ts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ts_addr);
    volatile uint32_t* clk = reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    {
        auto& parts_iface = get_local_cb_interface(cb_parts);
        const uint32_t parts = parts_iface.fifo_limit - parts_iface.fifo_size;
        for (uint32_t c = 0; c < children; c++) {
            const uint32_t cx = get_arg_val<uint32_t>(9 + 2 * c);
            const uint32_t cy = get_arg_val<uint32_t>(10 + 2 * c);
            noc_inline_dw_write(get_noc_addr(cx, cy, get_semaphore(sem_addr)), parts);
        }
    }
    volatile tt_l1_ptr uint32_t* addr_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_addr));
    while (*addr_sem == 0) {
    }
    const uint32_t part_dst = *addr_sem + slot * kPart * kTile;
    const uint64_t m_sem = get_noc_addr(lx, ly, get_semaphore(sem_m));
    const uint64_t part_sem = get_noc_addr(px, py, get_semaphore(sem_part));
    const uint32_t out_cb = children > 0 ? cb_osum : cb_o;
    for (uint32_t l = 0; l < layers; l++) {
        cb_wait_front(cb_m, 1);
        noc_async_write(get_read_ptr(cb_m), m_dst, heads * kFaceRow);
        noc_async_write_barrier();
        noc_semaphore_inc(m_sem, 1);
        cb_pop_front(cb_m, 1);
        cb_wait_front(out_cb, kPart);
        const uint32_t src = get_read_ptr(out_cb);
        for (uint32_t d = 0; d < kPart; d++) {
            noc_async_write(src + d * kTile, get_noc_addr(px, py, part_dst + d * kTile), 2 * kFace);
        }
        noc_async_write_barrier();
        noc_semaphore_inc(part_sem, 1);
        cb_pop_front(out_cb, kPart);
        if (ts_addr) {
            ts[l * 4 + 3] = *clk;
        }
        if (role == 2) {
            const uint32_t set = l % weight_sets;
            const uint32_t bank = get_arg_val<uint32_t>(tail_args);
            const uint32_t row = get_arg_val<uint32_t>(tail_args + 1);
            cb_wait_front(cb_flush, 2 * Dt);
            for (uint32_t kv = 0; kv < 2; kv++) {
                const uint32_t addr = get_arg_val<uint32_t>(tail_args + 2 + 2 * set + kv) + row * kKvRow;
                noc_async_write(
                    get_read_ptr(cb_flush) + kv * kKvRow, get_noc_addr_from_bank_id<true>(bank, addr), kKvRow);
            }
            noc_async_write_barrier();
            cb_pop_front(cb_flush, 2 * Dt);
        }
    }
    noc_async_atomic_barrier();
}
