// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Resident streamer writer (BRISC): the activation plumbing of one streamer core. Per layer:
//   mixer half: release the hub's partial slots; send this core's projection columns to every mixer
//     core of the layer (GDN: the head cores; attention: the leader and the tail core) and write a GDN
//     layer's newest conv input back to its history slot (the oldest one, see conv_step); release o once the
//     mixer cores have written it; send the out-proj columns to the hub.
//   mlp half: release the slots; exchange silu(g) * u slices with every streamer; send the down
//     columns to the hub.
// Every buffer is rewritten only after a dependency chain through this core's use of it, so none is
// double-buffered (e.g. a head's next o needs this core's next projection row, which needs the mlp
// slots, which need this core's down output, which comes after its out-proj used the current o).
//
// Runtime args: 0 ng, 1 act_off, 2 nd, 3 pout_off, 4 hub_x, 5 hub_y, 6 hub_slots, 7 act_addr,
// 8 GDN projection tiles, 9 their tile offset in the GDN row, 10 attention projection tiles, 11 their
// offset in the attention row, 12 mixer row buffer address (same on every mixer core), 13 n_conv,
// 14 conv history address, 15 this core's history rows base, 16 GDN copies, 17 token state address,
// then gdn_heads x (x, y) of the head cores, then 2 x (x, y) of the attention leader and tail core,
// then num_streamers x (x, y) of the streamers, then an optional timeline buffer (0 = off): per layer 8
// wall-clock words at the phase boundaries (slots, row ready, mixer done, attn partial sent, slots,
// act slice ready, act complete, mlp partial sent), after the reader's 8 words.

#include "api/dataflow/dataflow_api.h"
#include "streamer_common.hpp"

using namespace resident;

void kernel_main() {
    const uint32_t ng = get_arg_val<uint32_t>(0);
    const uint32_t act_off = get_arg_val<uint32_t>(1);
    const uint32_t nd = get_arg_val<uint32_t>(2);
    const uint32_t pout_off = get_arg_val<uint32_t>(3);
    const uint32_t hub_x = get_arg_val<uint32_t>(4);
    const uint32_t hub_y = get_arg_val<uint32_t>(5);
    const uint32_t hub_slots = get_arg_val<uint32_t>(6);
    const uint32_t act_addr = get_arg_val<uint32_t>(7);
    const uint32_t nq_gdn = get_arg_val<uint32_t>(8);
    const uint32_t q_off_gdn = get_arg_val<uint32_t>(9);
    const uint32_t nq_attn = get_arg_val<uint32_t>(10);
    const uint32_t q_off_attn = get_arg_val<uint32_t>(11);
    const uint32_t mixer_rows = get_arg_val<uint32_t>(12);
    const uint32_t n_conv = get_arg_val<uint32_t>(13);
    const uint32_t hist_addr = get_arg_val<uint32_t>(14);
    const uint32_t first_row = get_arg_val<uint32_t>(15);
    const uint32_t gdn_copies = get_arg_val<uint32_t>(16);
    const uint32_t tok_addr = get_arg_val<uint32_t>(17);
    constexpr uint32_t heads_base = 18;
    constexpr uint32_t attn_base = heads_base + 2 * gdn_heads;
    constexpr uint32_t peers_base = attn_base + 4;
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
    const InterleavedAddrGen<true> hist{.bank_base_address = hist_addr, .page_size = 3 * kBlkBytes};
    uint32_t pos;
    {
        // the position picks the history slot to overwrite; the o buffer is free until the first o
        const InterleavedAddrGen<true> tok{.bank_base_address = tok_addr, .page_size = kTokRope + 3 * 2048};
        const uint32_t scratch = get_write_ptr(cb_o_in);
        noc_async_read(tok.get_noc_addr(0), scratch, 16);
        noc_async_read_barrier();
        pos = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    }

    // round 0's slots (x0) are seeded and released by the reader
    auto take_slots = [&](uint32_t round) {
        if (round > 0) {
            noc_semaphore_wait_min(slots_sem, round);
            cb_reserve_back(cb_slots, num_chips * Ht);
            cb_push_back(cb_slots, num_chips * Ht);
        }
    };
    auto send_partial = [&](uint32_t round) {
        cb_wait_front(cb_pout, nd_max);
        const uint32_t dst = hub_slots + (((round & 1) * num_chips + chip) * Ht + pout_off) * kTileBytes;
        noc_async_write(get_read_ptr(cb_pout), get_noc_addr(hub_x, hub_y, dst), nd * kTileBytes);
        noc_async_write_barrier();
        noc_semaphore_inc(hub_gather_sem, 1);
        cb_pop_front(cb_pout, nd_max);
    };
    auto send_row = [&](uint32_t first, uint32_t count, uint32_t nq, uint32_t q_off) {
        const uint32_t row = get_read_ptr(cb_qkvz_out);
        for (uint32_t i = 0; i < count; i++) {
            const uint32_t hx = get_arg_val<uint32_t>(first + 2 * i);
            const uint32_t hy = get_arg_val<uint32_t>(first + 2 * i + 1);
            noc_async_write(row, get_noc_addr(hx, hy, mixer_rows + q_off * kTileBytes), nq * kTileBytes);
        }
        noc_async_write_barrier();
        for (uint32_t i = 0; i < count; i++) {
            const uint32_t hx = get_arg_val<uint32_t>(first + 2 * i);
            const uint32_t hy = get_arg_val<uint32_t>(first + 2 * i + 1);
            noc_semaphore_inc(get_noc_addr(hx, hy, rows_sem_addr), 1);
        }
    };

    uint32_t g = 0;
    uint32_t o_expected = 0;  // o writes so far: gdn_heads per GDN layer, the leader per attention layer
    for (uint32_t l = 0; l < layers; l++) {
        const bool attn = is_attn(l);
        // mixer half
        take_slots(2 * l);
        mark(l, 0);
        cb_wait_front(cb_qkvz_out, kBlk);
        mark(l, 1);
        if (attn) {
            send_row(attn_base, 2, nq_attn, q_off_attn);
            o_expected += 1;
        } else {
            send_row(heads_base, gdn_heads, nq_gdn, q_off_gdn);
            o_expected += gdn_heads;
            cb_wait_front(cb_hist_out, 1);
            if (n_conv > 0) {
                noc_async_write(
                    get_read_ptr(cb_hist_out),
                    hist.get_noc_addr(first_row + g % gdn_copies, (conv_step(pos, g, gdn_copies) % 3) * kBlkBytes),
                    kBlkBytes);
                noc_async_write_barrier();
            }
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_local)) = g + 1;
            cb_pop_front(cb_hist_out, 1);
            g++;
        }
        cb_pop_front(cb_qkvz_out, kBlk);
        noc_semaphore_wait_min(heads_sem, o_expected);
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
    if constexpr (lm_head) {
        take_slots(2 * layers);
    }
    noc_async_atomic_barrier();
}
