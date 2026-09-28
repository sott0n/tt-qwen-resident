// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Resident GDN head core writer (BRISC), one value head h. Per GDN layer g (copy g % copies): packs
// row 0 of the gated output tiles into 1x32 tiles and writes them into every streamer's mixer-output
// buffer at this head's offset (bumping the streamers' head semaphore), then stores the updated
// recurrent state back to DRAM, off the critical path.
//
// Compile-time args: 0 Kt, 1 Vt, 2 GDN layers, 3 copies, 4 num_streamers, 5 sem_heads, 6 heads (per chip),
//   7 sem_state (local: state write-backs done)
// Runtime args: 0 state address (DRAM, see the reader), 1 o_in_addr (streamers' buffer), 2 head index,
// then num_streamers x (x, y), then an optional timeline buffer (0 = off; per layer [2] state out,
// [3] output sent)

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t Kt = get_compile_time_arg_val(0);
constexpr uint32_t Vt = get_compile_time_arg_val(1);
constexpr uint32_t layers = get_compile_time_arg_val(2);
constexpr uint32_t copies = get_compile_time_arg_val(3);
constexpr uint32_t num_streamers = get_compile_time_arg_val(4);
constexpr uint32_t sem_heads = get_compile_time_arg_val(5);
constexpr uint32_t num_heads = get_compile_time_arg_val(6);
constexpr uint32_t sem_state = get_compile_time_arg_val(7);
constexpr uint32_t st = Kt * Vt;
constexpr uint32_t cb_s_new_out = 24, cb_out = 28, cb_stage = 22;
constexpr uint32_t kBf16Tile = 2048, kF32Tile = 4096, kTiny = 64, kFaceBytes = 512;

void kernel_main() {
    const uint32_t state = get_arg_val<uint32_t>(0);
    const uint32_t o_in = get_arg_val<uint32_t>(1);
    const uint32_t head = get_arg_val<uint32_t>(2);
    constexpr uint32_t peers_base = 3;
    const uint32_t heads_sem_addr = get_semaphore(sem_heads);
    const InterleavedAddrGenFast<true> state_dram{
        .bank_base_address = state, .page_size = kF32Tile, .data_format = DataFormat::Float32};
    const uint32_t ts_addr = get_arg_val<uint32_t>(peers_base + 2 * num_streamers);
    volatile tt_l1_ptr uint32_t* ts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ts_addr);
    volatile uint32_t* clk = reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    auto& stage_iface = get_local_cb_interface(cb_stage);
    const uint32_t stage = stage_iface.fifo_limit - stage_iface.fifo_size;

    for (uint32_t l = 0; l < layers; l++) {
        const uint32_t set = l % copies;
        cb_wait_front(cb_out, Vt);
        const uint32_t out = get_read_ptr(cb_out);
        for (uint32_t t = 0; t < Vt; t++) {
            noc_async_read(get_noc_addr(out + t * kBf16Tile), stage + t * kTiny, kTiny / 2);
            noc_async_read(get_noc_addr(out + t * kBf16Tile + kFaceBytes), stage + t * kTiny + kTiny / 2, kTiny / 2);
        }
        noc_async_read_barrier();
        cb_pop_front(cb_out, Vt);
        for (uint32_t s = 0; s < num_streamers; s++) {
            const uint32_t px = get_arg_val<uint32_t>(peers_base + 2 * s);
            const uint32_t py = get_arg_val<uint32_t>(peers_base + 2 * s + 1);
            noc_async_write(stage, get_noc_addr(px, py, o_in + head * Vt * kTiny), Vt * kTiny);
        }
        noc_async_write_barrier();
        for (uint32_t s = 0; s < num_streamers; s++) {
            const uint32_t px = get_arg_val<uint32_t>(peers_base + 2 * s);
            const uint32_t py = get_arg_val<uint32_t>(peers_base + 2 * s + 1);
            noc_semaphore_inc(get_noc_addr(px, py, heads_sem_addr), 1);
        }
        cb_wait_front(cb_s_new_out, st);
        if (ts_addr) {
            ts[l * 4 + 2] = *clk;
        }
        const uint32_t first = (set * num_heads + head) * st;
        for (uint32_t t = 0; t < st; t++) {
            noc_async_write_page(first + t, state_dram, get_read_ptr(cb_s_new_out) + t * kF32Tile);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_s_new_out, st);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_state)) = l + 1;

        if (ts_addr) {
            ts[l * 4 + 3] = *clk;
        }
    }
    noc_async_atomic_barrier();
}
