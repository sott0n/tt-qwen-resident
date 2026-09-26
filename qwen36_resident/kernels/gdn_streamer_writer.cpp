// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Resident GDN-layer streamer writer (BRISC): the activation plumbing of one streamer core. Per layer:
//   attention half: release the hub's partial slots; send this core's conv'd qkvzab columns to every
//     head core's row buffer; release the head outputs (o) once all heads have written theirs; send
//     the out-proj columns to the hub.
//   mlp half: release the slots; exchange silu(g) * u slices with every streamer; send the down
//     columns to the hub.
// Every buffer is rewritten only after a dependency chain through this core's use of it, so none is
// double-buffered (e.g. a head's next o needs this core's next qkvzab row, which needs the mlp slots,
// which need this core's down output, which comes after its out-proj used the current o).
//
// Runtime args: 0 ng, 1 act_off, 2 nd, 3 pout_off, 4 hub_x, 5 hub_y, 6 hub_slots, 7 act_addr, 8 nq,
// 9 q_off (tile offset of this core's qkvzab columns), 10 head_rows_addr, then num_heads x (x, y) of the
// head cores, then num_streamers x (x, y) of the streamers.
// Extra compile-time arg: 41 sem_rows (head cores' row-arrival semaphore).
// Last runtime arg: optional timeline buffer (0 = off): per layer 8 wall-clock words at the phase
// boundaries (slots, qkvzab row ready, heads done, attn partial sent, slots, act slice ready, act
// complete, mlp partial sent), after the reader's 8 words.

#include "api/dataflow/dataflow_api.h"
#include "gdn_layer_common.hpp"

using namespace resident_gdn;

constexpr uint32_t sem_rows = get_compile_time_arg_val(41);

void kernel_main() {
    const uint32_t ng = get_arg_val<uint32_t>(0);
    const uint32_t act_off = get_arg_val<uint32_t>(1);
    const uint32_t nd = get_arg_val<uint32_t>(2);
    const uint32_t pout_off = get_arg_val<uint32_t>(3);
    const uint32_t hub_x = get_arg_val<uint32_t>(4);
    const uint32_t hub_y = get_arg_val<uint32_t>(5);
    const uint32_t hub_slots = get_arg_val<uint32_t>(6);
    const uint32_t act_addr = get_arg_val<uint32_t>(7);
    const uint32_t nq = get_arg_val<uint32_t>(8);
    const uint32_t q_off = get_arg_val<uint32_t>(9);
    const uint32_t head_rows = get_arg_val<uint32_t>(10);
    constexpr uint32_t heads_base = 11;
    constexpr uint32_t peers_base = heads_base + 2 * num_heads;
    const uint32_t ts_addr = get_arg_val<uint32_t>(peers_base + 2 * num_streamers);
    volatile tt_l1_ptr uint32_t* ts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ts_addr);
    volatile uint32_t* clk = reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    auto mark = [&](uint32_t l, uint32_t i) {
        if (ts_addr) {
            ts[l * 16 + 8 + i] = *clk;
        }
    };

    volatile tt_l1_ptr uint32_t* slots_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_slots));
    volatile tt_l1_ptr uint32_t* act_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_act));
    volatile tt_l1_ptr uint32_t* heads_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_heads));
    const uint32_t act_sem_addr = get_semaphore(sem_act);
    const uint32_t rows_sem_addr = get_semaphore(sem_rows);
    const uint64_t hub_gather_sem = get_noc_addr(hub_x, hub_y, get_semaphore(sem_gather));

    auto take_slots = [&](uint32_t round) {
        if (round > 0) {
            noc_semaphore_wait_min(slots_sem, round);
        }
        cb_reserve_back(cb_slots, num_chips * Ht);
        cb_push_back(cb_slots, num_chips * Ht);
    };
    auto send_partial = [&](uint32_t round) {
        cb_wait_front(cb_pout, nd_max);
        const uint32_t dst = hub_slots + (((round & 1) * num_chips + chip) * Ht + pout_off) * kTileBytes;
        noc_async_write(get_read_ptr(cb_pout), get_noc_addr(hub_x, hub_y, dst), nd * kTileBytes);
        noc_async_write_barrier();
        noc_semaphore_inc(hub_gather_sem, 1);
        cb_pop_front(cb_pout, nd_max);
    };

    for (uint32_t l = 0; l < layers; l++) {
        // attention half
        take_slots(2 * l);
        mark(l, 0);
        cb_wait_front(cb_qkvz_out, kBlk);
        mark(l, 1);
        const uint32_t row = get_read_ptr(cb_qkvz_out);
        for (uint32_t i = 0; i < num_heads; i++) {
            const uint32_t hx = get_arg_val<uint32_t>(heads_base + 2 * i);
            const uint32_t hy = get_arg_val<uint32_t>(heads_base + 2 * i + 1);
            noc_async_write(row, get_noc_addr(hx, hy, head_rows + q_off * kTileBytes), nq * kTileBytes);
        }
        noc_async_write_barrier();
        for (uint32_t i = 0; i < num_heads; i++) {
            const uint32_t hx = get_arg_val<uint32_t>(heads_base + 2 * i);
            const uint32_t hy = get_arg_val<uint32_t>(heads_base + 2 * i + 1);
            noc_semaphore_inc(get_noc_addr(hx, hy, rows_sem_addr), 1);
        }
        cb_pop_front(cb_qkvz_out, kBlk);
        noc_semaphore_wait_min(heads_sem, num_heads * (l + 1));
        mark(l, 2);
        cb_reserve_back(cb_o_in, Ot);
        cb_push_back(cb_o_in, Ot);
        send_partial(2 * l);
        mark(l, 3);

        // mlp half
        take_slots(2 * l + 1);
        mark(l, 4);
        cb_wait_front(cb_aslice, kBlk);
        mark(l, 5);
        const uint32_t slice = get_read_ptr(cb_aslice);
        for (uint32_t s = 0; s < num_streamers; s++) {
            const uint32_t px = get_arg_val<uint32_t>(peers_base + 2 * s);
            const uint32_t py = get_arg_val<uint32_t>(peers_base + 2 * s + 1);
            noc_async_write(slice, get_noc_addr(px, py, act_addr + act_off * kTileBytes), ng * kTileBytes);
        }
        noc_async_write_barrier();
        for (uint32_t s = 0; s < num_streamers; s++) {
            const uint32_t px = get_arg_val<uint32_t>(peers_base + 2 * s);
            const uint32_t py = get_arg_val<uint32_t>(peers_base + 2 * s + 1);
            noc_semaphore_inc(get_noc_addr(px, py, act_sem_addr), 1);
        }
        cb_pop_front(cb_aslice, kBlk);
        noc_semaphore_wait_min(act_sem, num_streamers * (l + 1));
        mark(l, 6);
        cb_reserve_back(cb_act, It);
        cb_push_back(cb_act, It);
        send_partial(2 * l + 1);
        mark(l, 7);
    }
    noc_async_atomic_barrier();
}
