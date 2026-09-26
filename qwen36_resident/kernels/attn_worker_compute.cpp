// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Attention worker compute (see attn_common.hpp). Per layer (set = layer % weight_sets):
//   chunk worker: the partial attention of q over this core's KV chunk;
//   tail core: k = rope(rmsnorm_rows(k) * w_k[set]), the new k / v row into the tail tiles, the
//     partial over the tail tile (masked), then the tail tiles as bf8 for the DRAM write-back;
// a group head adds its children's partials to its own.
//
// Runtime args: 0 role (0 idle, 1 chunk worker, 2 tail core), 1 n_tiles (KV tiles of the chunk),
// 2 children.

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "attn_compute_common.hpp"

using namespace resident_attn;

void kernel_main() {
    const uint32_t role = get_arg_val<uint32_t>(0);
    const uint32_t n_tiles = get_arg_val<uint32_t>(1);
    const uint32_t children = get_arg_val<uint32_t>(2);
    if (role == 0) {
        return;
    }
    cb_wait_front(cb_one, 1);
    if (role == 1) {
        compute_kernel_hw_startup<SrcOrder::Reverse>(cb_q, cb_k, cb_s);
        for (uint32_t l = 0; l < layers; l++) {
            cb_wait_front(cb_k, chunk_max * Dt);
            cb_wait_front(cb_v, chunk_max * Dt);
            cb_wait_front(cb_q, Dt);
            scores(cb_q, cb_k, 0, n_tiles, false);
            row_max(n_tiles);
            probs_and_partial(cb_v, 0, n_tiles);
            if (children > 0) {
                sum_partials<true>(children);
            }
            cb_pop_front(cb_q, Dt);
            cb_pop_front(cb_k, chunk_max * Dt);
            cb_pop_front(cb_v, chunk_max * Dt);
        }
        return;
    }
    compute_kernel_hw_startup(cb_kraw, cb_kraw, cb_sq);
    cb_wait_front(cb_mean, 1);
    cb_wait_front(cb_kw, weight_sets * Dt);
    cb_wait_front(cb_rope, 3);
    cb_wait_front(cb_mask, 1);
    cb_wait_front(cb_rowsel, 2);
    for (uint32_t l = 0; l < layers; l++) {
        const uint32_t set = l % weight_sets;
        norm_rope(cb_kraw, cb_kw, set * Dt, cb_knew, cb_knew);
        update_tail();
        cb_wait_front(cb_q, Dt);
        cb_wait_front(cb_tailk, Dt);
        cb_wait_front(cb_tailv, Dt);
        scores(cb_q, cb_tailk, 0, 1, true);
        row_max(1);
        probs_and_partial(cb_tailv, 0, 1);
        if (children > 0) {
            sum_partials<true>(children);
        }
        cb_pop_front(cb_q, Dt);
        pack_tail_bf8();
    }
}
