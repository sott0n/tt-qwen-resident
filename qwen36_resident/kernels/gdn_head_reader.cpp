// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Resident GDN head core reader (NCRISC), one value head h (key head kh = h / group). Per GDN layer
// g (copy c = g % copies), once every streamer has written its qkvzab columns into this core's row
// buffer (1x32 tiles, conv already applied):
//   - q_kh, k_kh, v_h, z_h are placed in row 0 of 32x32 bf16 tiles (rows 1..31 stay zero),
//   - a_h / b_h and the copy's dt_bias_h / neg_exp_A_h are broadcast into full fp32 tiles,
//   - the copy's norm weight and this head's recurrent state are read from DRAM (ahead of the rows;
//     the writer stores the updated state back, and a state is read again only in the next step).
// The ones and row-0 mask constants are built once.
//
// Compile-time args: 0 Kt, 1 Vt, 2 GDN layers, 3 copies, 4 num_streamers, 5 sem_rows, 6 heads (per chip),
//   7 sem_state (count of state write-backs done by the writer; a copy's state is re-read within a step
//   only after its previous write-back)
// Runtime args: 0 rows_addr, 1 state address (DRAM, [copies][heads][Kt * Vt] fp32 tiles), 2 norm weight
//   address (DRAM, [copies][Vt] bf16 row-0 tiles), 3 q_tile, 4 k_tile, 5 v_tile, 6 z_tile, 7 a_tile,
//   8 a_elem, 9 b_tile, 10 b_elem, 11 head index, then copies x (dt_bias_h bits, neg_exp_A_h bits), then an
//   optional timeline buffer (0 = off): per layer 4 words, the reader writes [0] rows arrived, [1] inputs
//   pushed

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t Kt = get_compile_time_arg_val(0);
constexpr uint32_t Vt = get_compile_time_arg_val(1);
constexpr uint32_t layers = get_compile_time_arg_val(2);
constexpr uint32_t copies = get_compile_time_arg_val(3);
constexpr uint32_t num_streamers = get_compile_time_arg_val(4);
constexpr uint32_t sem_rows = get_compile_time_arg_val(5);
constexpr uint32_t num_heads = get_compile_time_arg_val(6);
constexpr uint32_t sem_state = get_compile_time_arg_val(7);
constexpr uint32_t st = Kt * Vt;

constexpr uint32_t cb_q_in = 0, cb_k_in = 1, cb_v_in = 2, cb_z_in = 3, cb_norm_w = 4, cb_s_in = 5;
constexpr uint32_t cb_a_full = 6, cb_b_full = 7, cb_ones = 8, cb_row_mask = 9, cb_s_mm = 17;
constexpr uint32_t cb_neg_a_full = 30, cb_dt_bias_full = 31;
constexpr uint32_t kBf16Tile = 2048, kF32Tile = 4096, kTiny = 64, kFaceBytes = 512;

inline uint32_t row0_offset(uint32_t c) { return c < 16 ? c : 256 + (c - 16); }

void fill_f32_tile(uint32_t l1_addr, uint32_t bits) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_addr);
    for (uint32_t i = 0; i < 16; i++) {
        p[i] = bits;
    }
    for (uint32_t bytes = 64; bytes < kF32Tile; bytes *= 2) {
        noc_async_read(get_noc_addr(l1_addr), l1_addr + bytes, bytes);
        noc_async_read_barrier();
    }
}

void zero_bytes(uint32_t l1_addr, uint32_t bytes) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_addr);
    for (uint32_t i = 0; i < bytes / 4; i++) {
        p[i] = 0;
    }
}

// n 1x32 row tiles starting at row tile `first` -> row 0 of n 32x32 bf16 tiles of `cb`
void place_rows(uint32_t cb, uint32_t rows, uint32_t first, uint32_t n) {
    cb_reserve_back(cb, n);
    const uint32_t dst = get_write_ptr(cb);
    for (uint32_t t = 0; t < n; t++) {
        const uint32_t src = rows + (first + t) * kTiny;
        noc_async_read(get_noc_addr(src), dst + t * kBf16Tile, kTiny / 2);
        noc_async_read(get_noc_addr(src + kTiny / 2), dst + t * kBf16Tile + kFaceBytes, kTiny / 2);
    }
    noc_async_read_barrier();
    cb_push_back(cb, n);
}

void push_fill(uint32_t cb, uint32_t bits) {
    cb_reserve_back(cb, 1);
    fill_f32_tile(get_write_ptr(cb), bits);
    cb_push_back(cb, 1);
}

// tiles first .. first + n of an interleaved DRAM tensor
template <typename AddrGen>
void push_pages(uint32_t cb, const AddrGen& src, uint32_t first, uint32_t n, uint32_t tile_bytes) {
    cb_reserve_back(cb, n);
    for (uint32_t t = 0; t < n; t++) {
        noc_async_read_page(first + t, src, get_write_ptr(cb) + t * tile_bytes);
    }
    noc_async_read_barrier();
    cb_push_back(cb, n);
}

}  // namespace

void kernel_main() {
    const uint32_t rows = get_arg_val<uint32_t>(0);
    const uint32_t state = get_arg_val<uint32_t>(1);
    const uint32_t norm_w = get_arg_val<uint32_t>(2);
    const uint32_t q_tile = get_arg_val<uint32_t>(3);
    const uint32_t k_tile = get_arg_val<uint32_t>(4);
    const uint32_t v_tile = get_arg_val<uint32_t>(5);
    const uint32_t z_tile = get_arg_val<uint32_t>(6);
    const uint32_t a_tile = get_arg_val<uint32_t>(7);
    const uint32_t a_elem = get_arg_val<uint32_t>(8);
    const uint32_t b_tile = get_arg_val<uint32_t>(9);
    const uint32_t b_elem = get_arg_val<uint32_t>(10);
    const uint32_t head = get_arg_val<uint32_t>(11);
    constexpr uint32_t gates_base = 12;
    const uint32_t ts_addr = get_arg_val<uint32_t>(gates_base + 2 * copies);
    const InterleavedAddrGenFast<true> state_dram{
        .bank_base_address = state, .page_size = kF32Tile, .data_format = DataFormat::Float32};
    const InterleavedAddrGenFast<true> norm_dram{
        .bank_base_address = norm_w, .page_size = kBf16Tile, .data_format = DataFormat::Float16_b};
    volatile tt_l1_ptr uint32_t* ts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ts_addr);
    volatile uint32_t* clk = reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);

    // Rows 1..31 of the row-0 input tiles are never written again: zero them once (the rank-1 update
    // multiplies them by zero, so they must be finite).
    for (uint32_t cb : {cb_q_in, cb_k_in, cb_v_in, cb_z_in, cb_norm_w}) {
        auto& iface = get_local_cb_interface(cb);
        zero_bytes(iface.fifo_limit - iface.fifo_size, iface.fifo_size);
    }
    push_fill(cb_ones, 0x3f800000u);
    cb_reserve_back(cb_row_mask, 1);
    {
        const uint32_t base = get_write_ptr(cb_row_mask);
        fill_f32_tile(base, 0u);
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
        for (uint32_t c = 0; c < 32; c++) {
            p[row0_offset(c)] = 0x3f800000u;
        }
    }
    cb_push_back(cb_row_mask, 1);

    volatile tt_l1_ptr uint32_t* rows_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_rows));
    volatile tt_l1_ptr uint32_t* state_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_state));
    volatile tt_l1_ptr uint16_t* a_row = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(rows + a_tile * kTiny);
    volatile tt_l1_ptr uint16_t* b_row = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(rows + b_tile * kTiny);
    for (uint32_t l = 0; l < layers; l++) {
        const uint32_t set = l % copies;
        // the layer's state and norm weight do not depend on the rows: land them first
        if (l >= copies) {
            noc_semaphore_wait_min(state_sem, l - copies + 1);
        }
        const uint32_t first = (set * num_heads + head) * st;
        cb_reserve_back(cb_s_in, st);
        push_pages(cb_s_mm, state_dram, first, st, kF32Tile);  // then a local copy: one DRAM read per state
        noc_async_read(get_noc_addr(get_read_ptr(cb_s_mm)), get_write_ptr(cb_s_in), st * kF32Tile);
        noc_async_read_barrier();
        cb_push_back(cb_s_in, st);
        push_pages(cb_norm_w, norm_dram, set * Vt, Vt, kBf16Tile);
        noc_semaphore_wait_min(rows_sem, num_streamers * (l + 1));
        if (ts_addr) {
            ts[l * 4 + 0] = *clk;
        }
        place_rows(cb_q_in, rows, q_tile, Kt);
        place_rows(cb_k_in, rows, k_tile, Kt);
        // a/b are the only scalars read by the RISC: the row buffer is written by other cores
        invalidate_l1_cache();
        const uint32_t a_bits = static_cast<uint32_t>(a_row[a_elem]) << 16;
        const uint32_t b_bits = static_cast<uint32_t>(b_row[b_elem]) << 16;
        push_fill(cb_a_full, a_bits);
        push_fill(cb_dt_bias_full, get_arg_val<uint32_t>(gates_base + 2 * set));
        push_fill(cb_neg_a_full, get_arg_val<uint32_t>(gates_base + 2 * set + 1));
        push_fill(cb_b_full, b_bits);
        place_rows(cb_v_in, rows, v_tile, Vt);
        place_rows(cb_z_in, rows, z_tile, Vt);
        if (ts_addr) {
            ts[l * 4 + 1] = *clk;
        }
    }
}
