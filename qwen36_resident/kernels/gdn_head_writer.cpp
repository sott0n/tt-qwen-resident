// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Resident GDN head core writer (BRISC), one value head h. Per layer (set = layer % weight_sets):
// packs row 0 of the gated output tiles into 1x32 tiles and writes them into every streamer's
// attention-output buffer at this head's offset (bumping the streamers' head semaphore), then stores
// the updated recurrent state back into the set's slot in L1, off the critical path.
//
// Compile-time args: 0 Kt, 1 Vt, 2 layers, 3 weight_sets, 4 num_streamers, 5 sem_heads, 6 sem_state
// Runtime args: 0 state_addr, 1 o_in_addr (streamers' buffer), 2 head index, then num_streamers x (x, y),
// then an optional timeline buffer (0 = off; per layer [2] state out, [3] output sent)

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t Kt = get_compile_time_arg_val(0);
constexpr uint32_t Vt = get_compile_time_arg_val(1);
constexpr uint32_t layers = get_compile_time_arg_val(2);
constexpr uint32_t weight_sets = get_compile_time_arg_val(3);
constexpr uint32_t num_streamers = get_compile_time_arg_val(4);
constexpr uint32_t sem_heads = get_compile_time_arg_val(5);
constexpr uint32_t sem_state = get_compile_time_arg_val(6);
constexpr uint32_t st = Kt * Vt;
constexpr uint32_t cb_s_new_out = 24, cb_out = 28, cb_stage = 22;
constexpr uint32_t kBf16Tile = 2048, kF32Tile = 4096, kTiny = 64, kFaceBytes = 512;

void kernel_main() {
    const uint32_t state = get_arg_val<uint32_t>(0);
    const uint32_t o_in = get_arg_val<uint32_t>(1);
    const uint32_t head = get_arg_val<uint32_t>(2);
    constexpr uint32_t peers_base = 3;
    const uint32_t heads_sem_addr = get_semaphore(sem_heads);
    volatile tt_l1_ptr uint32_t* state_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_state));
    const uint32_t ts_addr = get_arg_val<uint32_t>(peers_base + 2 * num_streamers);
    volatile tt_l1_ptr uint32_t* ts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ts_addr);
    volatile uint32_t* clk = reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    auto& stage_iface = get_local_cb_interface(cb_stage);
    const uint32_t stage = stage_iface.fifo_limit - stage_iface.fifo_size;

    for (uint32_t l = 0; l < layers; l++) {
        const uint32_t set = l % weight_sets;
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
        noc_async_write(get_read_ptr(cb_s_new_out), get_noc_addr(state + set * st * kF32Tile), st * kF32Tile);
        noc_async_write_barrier();
        cb_pop_front(cb_s_new_out, st);
        *state_sem = l + 1;

        if (ts_addr) {
            ts[l * 4 + 3] = *clk;
        }
    }
    noc_async_atomic_barrier();
}
