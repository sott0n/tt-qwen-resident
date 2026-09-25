// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Resident MLP streamer reader (NCRISC): streams this core's gate|up and down columns of every layer
// from its DRAM bank into the shared weight ring, running ahead of compute by up to the ring size.
//
// Runtime args: 0 bank_id, 1 vc, then per entry e: 2 + 2e n_tiles, 3 + 2e byte offset of this core's
// first column inside the bank shard; then weight_sets * kEntries weight base addresses (set-major).

#include "api/dataflow/dataflow_api.h"
#include "mlp_common.hpp"

using namespace resident_mlp;

namespace {

constexpr uint32_t kInFlight = 2;

uint32_t ring_base;
volatile uint32_t* consumed;
RingCursor cursor;

FORCE_INLINE void wait_for_room() {
    constexpr uint32_t ring_units = ring_bytes / kUnit;
    while (((cursor.units - *consumed) & 0xffff) > ring_units) {
    }
}

template <uint32_t E>
FORCE_INLINE void stream_entry(uint32_t bank_id, uint32_t vc, uint32_t weight_addr) {
    using En = Entry<E>;
    const uint32_t n_tiles = get_arg_val<uint32_t>(2 + 2 * E);
    const uint32_t col_offset = get_arg_val<uint32_t>(3 + 2 * E);
    const uint32_t num_blocks = n_tiles * En::nkb;
    const uint64_t base = get_noc_addr_from_bank_id<true>(bank_id, weight_addr);
    noc_async_read_one_packet_set_state<true>(base, En::page_size, vc);

    uint32_t src = col_offset;
    uint32_t issued = 0;
    uint32_t next_trid = 1;
    uint32_t wait_trid = 1;
    for (uint32_t b = 0; b < num_blocks; b++) {
        uint32_t l1 = ring_base + cursor.place<E>();
        wait_for_room();
        noc_async_read_set_trid(next_trid);
        for (uint32_t p = 0; p < En::pages; p++) {
            noc_async_read_one_packet_with_state_with_trid(base, src, l1, next_trid);
            src += En::page_size;
            l1 += En::page_size;
        }
        next_trid = next_trid == kInFlight + 1 ? 1 : next_trid + 1;
        if (++issued == kInFlight + 1) {
            noc_async_read_barrier_with_trid(wait_trid);
            cb_push_back(En::cb, En::sb);
            wait_trid = wait_trid == kInFlight + 1 ? 1 : wait_trid + 1;
            issued--;
        }
    }
    while (issued > 0) {
        noc_async_read_barrier_with_trid(wait_trid);
        cb_push_back(En::cb, En::sb);
        wait_trid = wait_trid == kInFlight + 1 ? 1 : wait_trid + 1;
        issued--;
    }
}

}  // namespace

void kernel_main() {
    const uint32_t bank_id = get_arg_val<uint32_t>(0);
    const uint32_t vc = get_arg_val<uint32_t>(1);
    constexpr uint32_t addr_base = 2 + 2 * kEntries;
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    {
        auto& w = get_local_cb_interface(cb_w0);
        ring_base = w.fifo_limit - w.fifo_size;
        consumed = get_cb_tiles_acked_ptr(cb_consumed);
        *consumed = 0;
    }
    // rmsnorm weights are tensor-backed and resident for the whole run
    cb_reserve_back(cb_gamma, weight_sets * Ht);
    cb_push_back(cb_gamma, weight_sets * Ht);
    for (uint32_t l = 0; l < layers; l++) {
        const uint32_t set = l % weight_sets;
        stream_entry<0>(bank_id, vc, get_arg_val<uint32_t>(addr_base + set * kEntries + 0));
        stream_entry<1>(bank_id, vc, get_arg_val<uint32_t>(addr_base + set * kEntries + 1));
    }
}
