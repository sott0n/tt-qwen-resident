// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Attention worker compute (see attn_common.hpp). Per attention layer:
//   chunk worker: the partial attention of q over this core's KV chunk (a zero partial without one);
//   tail core: k = rope(rmsnorm_rows(k) * w_k), the new k / v row into the tail tiles, the partial over
//     the tail tile (masked), then the tail tiles as bf8 for the DRAM write-back;
// a group head adds its children's partials to its own.
//
// Runtime args: 0 role (1 chunk worker, 2 tail core), 1 children. The chunk's KV tiles come in cb_ntiles.

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/fill.h"
#include "attn_compute_common.hpp"

using namespace resident_attn;

namespace {

FORCE_INLINE void zero_partial() {
    cb_reserve_back(cb_o, kPart);
    pack_reconfig_data_format(cb_o);
    fill_tile_init();
    for (uint32_t d = 0; d < kPart; d++) {
        tile_regs_acquire();
        fill_tile(0, 0.0f);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_o);
        tile_regs_release();
    }
    cb_push_back(cb_o, kPart);
}

}  // namespace

void kernel_main() {
    const uint32_t role = get_arg_val<uint32_t>(0);
    const uint32_t children = get_arg_val<uint32_t>(1);
    cb_wait_front(cb_one, 1);
    cb_wait_front(cb_ntiles, 1);
    const uint32_t n_tiles = read_tile_value(cb_ntiles, 0, 0);
    if (role == 1) {
        compute_kernel_hw_startup<SrcOrder::Reverse>(cb_q, cb_k, cb_s);
        for (uint32_t l = 0; l < layers; l++) {
            cb_wait_front(cb_q, Dt);
            if (n_tiles > 0) {
                cb_wait_front(cb_k, chunk_max * Dt);
                cb_wait_front(cb_v, chunk_max * Dt);
                scores(cb_q, cb_k, 0, n_tiles, false);
                row_max(n_tiles);
                probs_and_partial(cb_v, 0, n_tiles);
                cb_pop_front(cb_k, chunk_max * Dt);
                cb_pop_front(cb_v, chunk_max * Dt);
            } else {
                cb_wait_front(cb_M, 1);
                zero_partial();
                cb_pop_front(cb_M, 1);
            }
            if (children > 0) {
                sum_partials<true>(children);
            }
            cb_pop_front(cb_q, Dt);
        }
        return;
    }
    compute_kernel_hw_startup(cb_kraw, cb_kraw, cb_sq);
    cb_wait_front(cb_mean, 1);
    cb_wait_front(cb_rope, 3);
    cb_wait_front(cb_mask, 1);
    cb_wait_front(cb_rowsel, 2);
    for (uint32_t l = 0; l < layers; l++) {
        cb_wait_front(cb_kw, Dt);
        norm_rope(cb_kraw, cb_kw, 0, cb_knew, cb_knew);
        cb_pop_front(cb_kw, Dt);
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
