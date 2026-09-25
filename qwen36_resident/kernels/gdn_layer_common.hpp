// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Resident Qwen3.6 linear-attention (Gated DeltaNet) decoder layer: shared compile-time schedule of
// the streamer cores. Per layer, two residual halves run back to back:
//   attention: h = rmsnorm(x) * gamma_in;  y = h @ W_qkvzab (this core's columns);
//              conv1d(4 taps) + silu on the q|k|v columns (state kept per core);
//              y goes to the head cores, which run the delta-rule recurrence + gated rmsnorm and
//              return o (1 x value_dim); partial = o @ W_out (this core's columns)
//   mlp:       h = rmsnorm(x) * gamma_post;  partial = down(silu(h @ G) * (h @ U))
// and after each half the hub all-reduces the chips' partials (chip 0 folds x in) and multicasts
// the per-chip slots back, so every streamer rebuilds x as the sum of the slots.
//
// Compile-time args (identical on the three RISCs of a streamer):
//   0 layers, 1 weight_sets, 2 num_chips, 3 num_streamers, 4 Ht, 5 It (mlp inter tiles per chip),
//   6 chip, 7 ring_bytes, 8 rms eps bits, 9 1/sqrt(hidden) bits, 10 sem_slots, 11 sem_act, 12 sem_gather,
//   13 + 5e (e = 0 qkvzab, 1 out, 2 gate|up, 3 down): Kt, sb, pages, page_size, block_bytes,
//   33 ng_max, 34 nd_max, 35 nq_max (qkvzab tiles per core), 36 conv_tiles (q|k|v tiles per chip),
//   37 Ot (value_dim tiles = out-proj K tiles), 38 num_heads, 39 sem_heads, 40 row_tiles (padded
//   qkvzab tiles per chip, the head cores' row buffer)
#pragma once

#include <stdint.h>

namespace resident_gdn {

constexpr uint32_t kEntries = 4;
constexpr uint32_t kQkvz = 0, kOut = 1, kGateUp = 2, kDown = 3;

// streamer CBs
constexpr uint32_t cb_slots = 0;
constexpr uint32_t cb_w0 = 1;  // weight ring, entry e -> cb_w0 + e (1..4)
constexpr uint32_t cb_x = 5;
constexpr uint32_t cb_h = 6;
constexpr uint32_t cb_consumed = 7;
constexpr uint32_t cb_act = 8;     // mlp intermediate activation (It)
constexpr uint32_t cb_dout = 9;    // out-proj / down columns before the chip-0 residual fold
constexpr uint32_t cb_pout = 10;   // partial sent to the hub
constexpr uint32_t cb_gamma = 11;  // weight_sets x 2 x Ht: [set][attn, mlp]
constexpr uint32_t cb_x_full = 12;
constexpr uint32_t cb_h_full = 13;
constexpr uint32_t cb_gamma_full = 14;
constexpr uint32_t cb_gu = 15;
constexpr uint32_t cb_aslice = 16;
constexpr uint32_t cb_qkvz = 17;       // qkvzab matmul columns of this core
constexpr uint32_t cb_qkvz_out = 18;   // after conv1d + silu (q|k|v columns) / passthrough (z, a, b)
constexpr uint32_t cb_conv_w = 19;     // weight_sets x nq_max x 4 taps
constexpr uint32_t cb_conv_hist = 20;  // weight_sets x 3 x nq_max previous inputs (ring of 3 per set)
constexpr uint32_t cb_o_in = 21;       // attention output of all heads (Ot), written by the head cores
constexpr uint32_t kUnit = 64;
constexpr uint32_t kTileBytes = 64;

constexpr uint32_t layers = get_compile_time_arg_val(0);
constexpr uint32_t weight_sets = get_compile_time_arg_val(1);
constexpr uint32_t num_chips = get_compile_time_arg_val(2);
constexpr uint32_t num_streamers = get_compile_time_arg_val(3);
constexpr uint32_t Ht = get_compile_time_arg_val(4);
constexpr uint32_t It = get_compile_time_arg_val(5);
constexpr uint32_t chip = get_compile_time_arg_val(6);
constexpr uint32_t ring_bytes = get_compile_time_arg_val(7);
constexpr uint32_t eps_bits = get_compile_time_arg_val(8);
constexpr uint32_t inv_sqrt_hidden_bits = get_compile_time_arg_val(9);
constexpr uint32_t sem_slots = get_compile_time_arg_val(10);
constexpr uint32_t sem_act = get_compile_time_arg_val(11);
constexpr uint32_t sem_gather = get_compile_time_arg_val(12);
constexpr uint32_t ng_max = get_compile_time_arg_val(33);
constexpr uint32_t nd_max = get_compile_time_arg_val(34);
constexpr uint32_t nq_max = get_compile_time_arg_val(35);
constexpr uint32_t conv_tiles = get_compile_time_arg_val(36);
constexpr uint32_t Ot = get_compile_time_arg_val(37);
constexpr uint32_t num_heads = get_compile_time_arg_val(38);
constexpr uint32_t sem_heads = get_compile_time_arg_val(39);
constexpr uint32_t row_tiles = get_compile_time_arg_val(40);
constexpr uint32_t Hf = Ht / 32;

template <uint32_t E>
struct Entry {
    static constexpr uint32_t Kt = get_compile_time_arg_val(13 + 5 * E + 0);
    static constexpr uint32_t sb = get_compile_time_arg_val(13 + 5 * E + 1);
    static constexpr uint32_t pages = get_compile_time_arg_val(13 + 5 * E + 2);
    static constexpr uint32_t page_size = get_compile_time_arg_val(13 + 5 * E + 3);
    static constexpr uint32_t block_bytes = get_compile_time_arg_val(13 + 5 * E + 4);
    static constexpr uint32_t nkb = Kt / sb;
    static constexpr uint32_t cb = cb_w0 + E;
};

struct RingCursor {
    uint32_t pos = 0;
    uint32_t units = 0;

    template <uint32_t E>
    uint32_t place() {
        constexpr uint32_t block = Entry<E>::block_bytes;
        if (pos + block > ring_bytes) {
            units += (ring_bytes - pos) / kUnit;
            pos = 0;
        }
        const uint32_t at = pos;
        pos += block;
        units += block / kUnit;
        return at;
    }
};

}  // namespace resident_gdn
