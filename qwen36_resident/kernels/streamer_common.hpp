// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Resident Qwen3.6 decode: shared compile-time schedule of the streamer cores. One launch runs one
// decode step through `layers` decoder layers (layer l is a full-attention layer when
// attn_interval > 0 and (l + 1) % attn_interval == 0, else a linear-attention (Gated DeltaNet) layer)
// and, optionally, the lm_head. Every layer is two residual halves:
//   mixer: h = rmsnorm(x) (the norm weight is folded into the weights);  y = h @ W_proj (this core's
//          columns; GDN: qkvzab, conv1d(4 taps) + silu on the q|k|v columns, the conv history kept in
//          DRAM; attention: q | gate | k | v);  y goes to the layer's mixer cores (GDN head cores or
//          the attention leader + tail core), which return o;  partial = o @ W_out
//   mlp:   h = rmsnorm(x);  partial = down(silu(h @ G) * (h @ U))
// After each half the hub all-reduces the chips' partials (chip 0 folds x in) and multicasts the
// per-chip slots back, so every streamer rebuilds x as the sum of the slots. The lm_head computes
// this core's logit columns of rmsnorm(x) @ W_head into an L1 buffer read by the host.
//
// Weights: one DRAM tensor per entry holding `distinct` layers of that kind at a fixed per-bank stride;
// the k-th layer of a kind uses copy k % distinct (benchmarks run many layers over few copies).
//
// Compile-time args (identical on the three RISCs of a streamer):
//   0 layers, 1 attn_interval, 2 num_chips, 3 num_streamers, 4 Ht, 5 It, 6 chip, 7 ring_bytes,
//   8 rms eps bits, 9 1/sqrt(hidden) bits, 10 sem_slots, 11 sem_act, 12 sem_gather, 13 sem_heads,
//   14 sem_rows, 15 ng_max, 16 nd_max, 17 nq_max (projection tiles per core, <= kBlk), 18 nv_max
//   (lm_head tiles per core), 19 conv_tiles (q|k|v tiles per chip), 20 Ot (out-proj K tiles),
//   21 gdn_heads, 22 lm_head (0 / 1), then per entry e (see kEntries): 23 + 5e Kt, sb, pages,
//   page_size, block_bytes, then 53 sem_local (conv history write-backs done, for the reader)
#pragma once

#include <stdint.h>

namespace resident {

constexpr uint32_t kEntries = 6;
constexpr uint32_t kQkvz = 0, kQkvg = 1, kOut = 2, kGateUp = 3, kDown = 4, kHead = 5;

// streamer CBs
constexpr uint32_t cb_slots = 0;
constexpr uint32_t cb_w0 = 1;  // weight ring, entry e -> cb_w0 + e (1..6)
constexpr uint32_t cb_consumed = 7;
constexpr uint32_t cb_act = 8;    // mlp intermediate activation (It)
constexpr uint32_t cb_dout = 9;   // out-proj / down columns before the chip-0 residual fold
constexpr uint32_t cb_pout = 10;  // partial sent to the hub
constexpr uint32_t cb_x = 11;
constexpr uint32_t cb_x_full = 12;
constexpr uint32_t cb_h = 13;
constexpr uint32_t cb_h_full = 14;
constexpr uint32_t cb_g = 15;  // gate columns of this core (one full-tile block of 1x32 tiles)
constexpr uint32_t cb_aslice = 16;
constexpr uint32_t cb_qkvz = 17;       // projection columns of this core
constexpr uint32_t cb_qkvz_out = 18;   // after conv1d + silu (q|k|v columns) / passthrough
constexpr uint32_t cb_conv_w = 19;     // this GDN layer's 4 taps x kBlk (streamed)
constexpr uint32_t cb_conv_hist = 20;  // this GDN layer's 3 previous inputs x kBlk, oldest first (streamed)
constexpr uint32_t cb_o_in = 21;       // mixer output (Ot), written by the mixer cores
constexpr uint32_t cb_u = 26;          // up columns of this core
constexpr uint32_t cb_hist_out = 30;   // this step's input to the conv (the newest history entry)
constexpr uint32_t cb_logits = 31;     // lm_head columns of this core (read by the host)
// The per-core column blocks of the elementwise phases (conv1d + silu, silu(g) * u) hold at most
// kBlk 1x32 tiles and are laid out on a 32x32 tile's worth of bytes, so the SFPU math runs once on
// the 32x32 views below instead of once per 1x32 tile (an SFPU op covers a full 32x32 tile).
constexpr uint32_t kBlk = 32;
constexpr uint32_t cb_qkvz_full = 22;
constexpr uint32_t cb_qkvz_out_full = 23;
constexpr uint32_t cb_conv_w_full = 24;     // [tap] full tiles
constexpr uint32_t cb_conv_hist_full = 25;  // [slot] full tiles, oldest first
constexpr uint32_t cb_g_full = 27;
constexpr uint32_t cb_u_full = 28;
constexpr uint32_t cb_aslice_full = 29;
constexpr uint32_t kUnit = 64;
constexpr uint32_t kTileBytes = 64;
constexpr uint32_t kBlkBytes = kBlk * kTileBytes;  // a 32x32 bf16 tile's worth of 1x32 tiles

// token state (DRAM, uint32 words): [0] position; x0 (hidden bf16 as 1x32 tiles) at kTokX0; the rope
// cos, sin, -sin tiles of the position (32x32 bf16, rows replicated) at kTokRope
constexpr uint32_t kTokX0 = 64;

constexpr uint32_t layers = get_compile_time_arg_val(0);
constexpr uint32_t attn_interval = get_compile_time_arg_val(1);
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
constexpr uint32_t sem_heads = get_compile_time_arg_val(13);
constexpr uint32_t sem_rows = get_compile_time_arg_val(14);
constexpr uint32_t ng_max = get_compile_time_arg_val(15);
constexpr uint32_t nd_max = get_compile_time_arg_val(16);
constexpr uint32_t nq_max = get_compile_time_arg_val(17);
constexpr uint32_t nv_max = get_compile_time_arg_val(18);
constexpr uint32_t conv_tiles = get_compile_time_arg_val(19);
constexpr uint32_t Ot = get_compile_time_arg_val(20);
constexpr uint32_t gdn_heads = get_compile_time_arg_val(21);
constexpr bool lm_head = get_compile_time_arg_val(22) != 0;
constexpr uint32_t Hf = Ht / 32;
constexpr uint32_t kTokRope = kTokX0 + Ht * kTileBytes;
static_assert(
    get_compile_time_arg_val(15) <= 32 && get_compile_time_arg_val(17) <= 32, "column blocks exceed one full tile");

constexpr uint32_t sem_local = get_compile_time_arg_val(53);
constexpr uint32_t gdn_layers = layers - (attn_interval > 0 ? layers / attn_interval : 0);

FORCE_INLINE bool is_attn(uint32_t l) { return attn_interval > 0 && (l + 1) % attn_interval == 0; }

// The conv history of a GDN copy is a ring of 3 slots advanced once per use: GDN layer g (use g / copies
// of copy g % copies within the step) at position pos is the ring's step pos * uses + g / copies; slot
// (step + j) % 3 holds the j-th oldest input and the step's input replaces slot step % 3.
FORCE_INLINE uint32_t conv_step(uint32_t pos, uint32_t g, uint32_t copies) {
    const uint32_t c = g % copies;
    const uint32_t uses = (gdn_layers - c + copies - 1) / copies;
    return pos * uses + g / copies;
}

template <uint32_t E>
struct Entry {
    static constexpr uint32_t Kt = get_compile_time_arg_val(23 + 5 * E + 0);
    static constexpr uint32_t sb = get_compile_time_arg_val(23 + 5 * E + 1);
    static constexpr uint32_t pages = get_compile_time_arg_val(23 + 5 * E + 2);
    static constexpr uint32_t page_size = get_compile_time_arg_val(23 + 5 * E + 3);
    static constexpr uint32_t block_bytes = get_compile_time_arg_val(23 + 5 * E + 4);
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

}  // namespace resident
