// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Resident GDN-layer streamer compute (TRISC). Per layer (set = layer % weight_sets):
//   attention half:
//     x = sum of the partial slots;  h = rmsnorm(x) * gamma[set][0]
//     y = h @ W_qkvzab (this core's columns);  q|k|v columns: y = silu(conv1d_4(y)), with the three
//     previous inputs kept per set (ring of 3);  z, a, b columns pass through
//     d = o @ W_out (this core's columns, o = all heads' outputs);  pout = d (+ x on chip 0)
//   mlp half:
//     x = sum of the slots;  h = rmsnorm(x) * gamma[set][1];  a = silu(h @ G) * (h @ U)
//     d = act @ D;  pout = d (+ x on chip 0)
//
// Runtime args: 0 ng, 1 nd, 2 pout_off, 3 nq, 4 n_conv (leading q|k|v columns among this core's nq)

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/experimental/pack_block.h"
#include "api/compute/experimental/mul_reduce_scalar.h"
#include "api/compute/experimental/add_rsqrt.h"
#include "api/compute/experimental/rmsnorm.h"
#include "gdn_layer_common.hpp"

using namespace resident_gdn;

namespace {

constexpr uint32_t kDst = 8;  // tiles per DST chunk for the eltwise phases

RingCursor cursor;
uint32_t ring_base;
uint32_t consumed_addr_word;

template <uint32_t E, typename F>
FORCE_INLINE void with_block(F fn) {
    using En = Entry<E>;
    const uint32_t at = cursor.place<E>();
    UNPACK((get_local_cb_interface(En::cb).fifo_rd_ptr = ring_base + (at >> cb_addr_shift)));
    cb_wait_front(En::cb, En::sb);
    fn();
    cb_pop_front(En::cb, En::sb);
    UNPACK((TT_SETDMAREG(0, cursor.units & 0xffff, 0, LO_16(4))));
    UNPACK((TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::UNPACK)));
    UNPACK((TT_STOREREG(4, consumed_addr_word)));
}

// out[n] = in0[1, K] @ W[K, n] for n_tiles columns streamed through ring entry E; columns from
// `split` on go to out_b (gate | up), each output CB takes one block of block_tiles.
template <uint32_t E>
FORCE_INLINE void matmul(
    uint32_t in0_cb, uint32_t out_cb, uint32_t n_tiles, uint32_t block_tiles, uint32_t out_b = 0, uint32_t split = 0) {
    using En = Entry<E>;
    const bool two = split > 0;
    cb_wait_front(in0_cb, En::Kt);
    reconfig_full_operand<SrcOrder::Reverse>(in0_cb, En::cb);
    custom_mm_block_init_short<false, true, false>(in0_cb, En::cb, out_cb);
    pack_reconfig_data_format<true>(out_cb);
    pack_block_contiguous_init(out_cb);
    cb_reserve_back(out_cb, block_tiles);
    if (two) {
        cb_reserve_back(out_b, block_tiles);
    }
    for (uint32_t n = 0; n < n_tiles; n++) {
        tile_regs_acquire();
        for (uint32_t kb = 0; kb + 1 < En::nkb; kb++) {
            with_block<E>([&] { custom_mm_block<false>(in0_cb, En::cb, kb * En::sb, 0, 0, En::sb); });
        }
        with_block<E>([&] { custom_mm_block<true>(in0_cb, En::cb, (En::nkb - 1) * En::sb, 0, 0, En::sb); });
        tile_regs_commit();
        tile_regs_wait();
        pack_block_contiguous(0, two && n >= split ? out_b : out_cb, 1);
        tile_regs_release();
    }
    cb_push_back(out_cb, block_tiles);
    if (two) {
        cb_push_back(out_b, block_tiles);
    }
    custom_mm_block_uninit<false>();
}

// cb_x = sum over chips of cb_slots
static_assert(num_chips == 1 || num_chips % 2 == 0, "slots are summed in pairs");
FORCE_INLINE void sum_slots() {
    cb_wait_front(cb_slots, num_chips * Ht);
    cb_reserve_back(cb_x, Ht);
    reconfig_full_operand(cb_slots, cb_slots);
    pack_init(cb_x);
    for (uint32_t t0 = 0; t0 < Ht; t0 += kDst) {
        const uint32_t n = t0 + kDst <= Ht ? kDst : Ht - t0;
        tile_regs_acquire();
        if constexpr (num_chips == 1) {
            copy_init(cb_slots);
            for (uint32_t d = 0; d < n; d++) {
                copy_tile(cb_slots, t0 + d, d);
            }
        } else {
            for (uint32_t c = 0; c < num_chips; c += 2) {
                add_init(cb_slots, cb_slots, c > 0 /* acc_to_dest */);
                for (uint32_t d = 0; d < n; d++) {
                    add_tiles(cb_slots, cb_slots, c * Ht + t0 + d, (c + 1) * Ht + t0 + d, d);
                }
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t d = 0; d < n; d++) {
            pack_tile(d, cb_x);
        }
        tile_regs_release();
    }
    cb_push_back(cb_x, Ht);
    cb_pop_front(cb_slots, num_chips * Ht);
}

// cb_h = x / rms(x) * gamma[g], computed on the 32x32 views of x, gamma and h
FORCE_INLINE void rmsnorm(uint32_t g) {
    static_assert(Hf >= 1 && Hf <= 8 && Hf * 32 == Ht, "hidden must be 1..8 full tiles");
    cb_wait_front(cb_x, Ht);
    cb_reserve_back(cb_h, Ht);
    reconfig_full_operand(cb_x_full, cb_x_full);
    pack_reconfig_data_format<true>(cb_h_full);
    pack_block_contiguous_init(cb_h_full);
    mul_reduce_scalar_init(cb_x_full, cb_x_full);
    add_rsqrt_tile_init();
    tile_regs_acquire();
    mul_reduce_scalar_tile<PoolType::SUM>(
        cb_x_full, cb_x_full, cb_h_full, Hf, __builtin_bit_cast(float, inv_sqrt_hidden_bits));
    mul_reduce_scalar_uninit();
    add_rsqrt_tile<false, VectorMode::RC_custom, 1>(0, eps_bits);
    rmsnorm_mul_bcast_scalar_reuse_tiles_init<Hf>(cb_x_full);
    rmsnorm_mul_bcast_scalar_reuse_tiles<Hf, true>(cb_x_full, 0, 0, 0);
    mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_gamma_full);
    for (uint32_t i = 0; i < Hf; i++) {
        mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_gamma_full, g * Hf + i, i);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_block_contiguous(0, cb_h_full, Hf);
    tile_regs_release();
    cb_push_back(cb_h_full, Hf);  // only wraps the view's write pointer back to the base
    cb_push_back(cb_h, Ht);
    pack_reconfig_data_format<true>(cb_h);
}

// cb_aslice[i] = silu(g[i]) * u[i]
FORCE_INLINE void silu_mul() {
    cb_wait_front(cb_g, kBlk);
    cb_wait_front(cb_u, kBlk);
    cb_reserve_back(cb_aslice, kBlk);
    reconfig_full_operand(cb_g_full, cb_u_full);
    pack_reconfig_data_format<true>(cb_aslice_full);
    pack_init(cb_aslice_full);
    tile_regs_acquire();
    copy_init(cb_g_full);
    copy_tile(cb_g_full, 0, 0);
    copy_tile(cb_u_full, 0, 1);
    silu_tile_init();
    silu_tile(0);
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_aslice_full);
    tile_regs_release();
    cb_push_back(cb_aslice_full, 1);  // wraps the view's write pointer back to the base
    cb_push_back(cb_aslice, kBlk);
    cb_pop_front(cb_g, kBlk);
    cb_pop_front(cb_u, kBlk);
}

// cb_pout = cb_dout + x[pout_off : pout_off + nd]
FORCE_INLINE void add_residual(uint32_t nd, uint32_t pout_off) {
    cb_wait_front(cb_dout, nd_max);
    cb_reserve_back(cb_pout, nd_max);
    reconfig_full_operand(cb_dout, cb_x);
    add_init(cb_dout, cb_x);
    pack_init(cb_pout);
    for (uint32_t t0 = 0; t0 < nd; t0 += kDst) {
        const uint32_t n = t0 + kDst <= nd ? kDst : nd - t0;
        tile_regs_acquire();
        for (uint32_t d = 0; d < n; d++) {
            add_tiles(cb_dout, cb_x, t0 + d, pout_off + t0 + d, d);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t d = 0; d < n; d++) {
            pack_tile(d, cb_pout);
        }
        tile_regs_release();
    }
    cb_push_back(cb_pout, nd_max);
    cb_pop_front(cb_dout, nd_max);
}

// q|k|v columns: cb_qkvz_out = silu(w0 * y[t-3] + w1 * y[t-2] + w2 * y[t-1] + w3 * y[t]) and y[t] replaces
// y[t-3] in the set's history ring; the first n_conv of this core's columns are q|k|v, the rest
// (z, a, b) are copied through. The conv runs once on the 32x32 views of the column block.
uint32_t conv_step[weight_sets];

FORCE_INLINE void conv_pass(uint32_t set, uint32_t nq, uint32_t n_conv) {
    cb_wait_front(cb_qkvz, kBlk);
    cb_reserve_back(cb_qkvz_out, kBlk);
    const uint32_t c = conv_step[set];
    const uint32_t oldest = set * 3 + c % 3;
    if (n_conv > 0) {
        reconfig_full_operand(cb_qkvz_full, cb_qkvz_full);
        pack_reconfig_data_format<true>(cb_qkvz_out_full);
        pack_init(cb_qkvz_out_full);
        tile_regs_acquire();
        copy_init(cb_qkvz_full);
        copy_tile(cb_qkvz_full, 0, 0);
        copy_tile(cb_conv_w_full, set * 4 + 3, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 1);
        for (uint32_t j = 0; j < 3; j++) {
            copy_init(cb_qkvz_full);
            copy_tile(cb_conv_hist_full, set * 3 + (c + j) % 3, 2);
            copy_tile(cb_conv_w_full, set * 4 + j, 3);
            mul_binary_tile_init();
            mul_binary_tile(2, 3, 2);
            add_binary_tile_init();
            add_binary_tile(1, 2, 1);
        }
        silu_tile_init();
        silu_tile(1);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(1, cb_qkvz_out_full);
        pack_tile<true>(0, cb_conv_hist_full, oldest);  // newest input overwrites the oldest
        tile_regs_release();
        cb_push_back(cb_qkvz_out_full, 1);  // wraps the view's write pointer back to the base
    }
    if (n_conv < nq) {
        reconfig_full_operand(cb_qkvz, cb_qkvz);
        pack_reconfig_data_format<true>(cb_qkvz_out);
        pack_init(cb_qkvz_out);
        copy_init(cb_qkvz);
        for (uint32_t i = n_conv; i < nq; i++) {
            tile_regs_acquire();
            copy_tile(cb_qkvz, i, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile<true>(0, cb_qkvz_out, i);
            tile_regs_release();
        }
    }
    pack_reconfig_data_format<true>(cb_qkvz_out);
    cb_push_back(cb_qkvz_out, kBlk);
    cb_pop_front(cb_qkvz, kBlk);
    conv_step[set] = c + 1;
}

// d = in0 @ W_E over this core's down-type columns; pout = d (+ x on chip 0)
template <uint32_t E>
FORCE_INLINE void project_to_partial(uint32_t in0_cb, uint32_t in0_tiles, uint32_t nd, uint32_t pout_off) {
    if constexpr (chip == 0) {
        matmul<E>(in0_cb, cb_dout, nd, nd_max);
        cb_pop_front(in0_cb, in0_tiles);
        add_residual(nd, pout_off);
    } else {
        matmul<E>(in0_cb, cb_pout, nd, nd_max);
        cb_pop_front(in0_cb, in0_tiles);
    }
}

}  // namespace

void kernel_main() {
    const uint32_t ng = get_arg_val<uint32_t>(0);
    const uint32_t nd = get_arg_val<uint32_t>(1);
    const uint32_t pout_off = get_arg_val<uint32_t>(2);
    const uint32_t nq = get_arg_val<uint32_t>(3);
    const uint32_t n_conv = get_arg_val<uint32_t>(4);
#ifdef TRISC_UNPACK
    {
        auto& w = get_local_cb_interface(cb_w0);
        ring_base = w.fifo_limit - w.fifo_size;
        consumed_addr_word = static_cast<uint32_t>(
            (reinterpret_cast<std::uintptr_t>(get_cb_tiles_acked_ptr(cb_consumed)) >> 2) & 0x3ffff);
    }
#endif
    for (uint32_t s = 0; s < weight_sets; s++) {
        conv_step[s] = 0;
    }
    custom_mm_block_init<false, true, false>(cb_h, Entry<0>::cb, cb_qkvz);
    custom_mm_block_uninit<false>();
    cb_wait_front(cb_gamma, weight_sets * 2 * Ht);
    cb_wait_front(cb_conv_w, weight_sets * 4 * kBlk);

    for (uint32_t l = 0; l < layers; l++) {
        const uint32_t set = l % weight_sets;
        // attention half
        sum_slots();
        rmsnorm(set * 2);
        matmul<kQkvz>(cb_h, cb_qkvz, nq, kBlk);
        cb_pop_front(cb_h, Ht);
        conv_pass(set, nq, n_conv);
        project_to_partial<kOut>(cb_o_in, Ot, nd, pout_off);
        cb_pop_front(cb_x, Ht);
        // mlp half
        sum_slots();
        rmsnorm(set * 2 + 1);
        matmul<kGateUp>(cb_h, cb_g, 2 * ng, kBlk, cb_u, ng);
        cb_pop_front(cb_h, Ht);
        silu_mul();
        project_to_partial<kDown>(cb_act, It, nd, pout_off);
        cb_pop_front(cb_x, Ht);
    }
}
