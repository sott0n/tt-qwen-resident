// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Resident MLP hub (BRISC, one core per chip): pure data movement between the streamers and the fabric.
// Per layer l (slot parity p = l & 1):
//   1. wait until every streamer has written its down columns into slot[p][chip] (gather semaphore),
//   2. line-multicast slot[p][chip] over fabric into slot[p][chip] of every peer hub (fused increment
//      of the peers' CCL semaphore), wait for the peers' slots,
//   3. multicast slot[p][0 .. num_chips) to the streamers' partial slots and bump their slots semaphore.
//      The multicast rectangle is the streamers' bounding box and contains the hub (loopback); the hub
//      and idle cores inside it only receive into addresses reserved for the streamers' buffers.
// The last layer's slots stay on the hub for the host to read back (and are also multicast when
// mcast_last is set, for a stage that runs after the last layer).
//
// Compile-time args: 0 num_chips, 1 chip, 2 vector_bytes, 3 packet_bytes, 4 layers, 5 num_streamers,
//   6 sem_gather id, 7 sem_slots id, 8..11 multicast rectangle (noc x0, y0, x1, y1), 12 multicast dests,
//   13 sem_flag id (local word holding the value multicast into the streamers' slots semaphore),
//   14 mcast_last
// Runtime args: 0 slots_addr, 1 ccl_sem_addr (global semaphore), 2 streamer slots addr, then FabricConnectionManager
// args (fwd, bwd)

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"

constexpr uint32_t num_chips = get_compile_time_arg_val(0);
constexpr uint32_t chip = get_compile_time_arg_val(1);
constexpr uint32_t vector_bytes = get_compile_time_arg_val(2);
constexpr uint32_t packet_bytes = get_compile_time_arg_val(3);
constexpr uint32_t layers = get_compile_time_arg_val(4);
constexpr uint32_t num_streamers = get_compile_time_arg_val(5);
constexpr uint32_t sem_gather = get_compile_time_arg_val(6);
constexpr uint32_t sem_slots = get_compile_time_arg_val(7);
constexpr uint32_t mc_x0 = get_compile_time_arg_val(8);
constexpr uint32_t mc_y0 = get_compile_time_arg_val(9);
constexpr uint32_t mc_x1 = get_compile_time_arg_val(10);
constexpr uint32_t mc_y1 = get_compile_time_arg_val(11);
constexpr uint32_t mc_dests = get_compile_time_arg_val(12);
constexpr bool mcast_last = get_compile_time_arg_val(14) != 0;
constexpr uint32_t sem_flag = get_compile_time_arg_val(13);
constexpr uint32_t packets_per_vector = (vector_bytes + packet_bytes - 1) / packet_bytes;

void kernel_main() {
    size_t arg_idx = 0;
    const uint32_t slots = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t ccl_sem_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t streamer_slots = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t flag_addr = get_semaphore(sem_flag);
    auto fabric =
        FabricConnectionManager::build_from_args<FabricConnectionManager::BUILD_AND_OPEN_CONNECTION_START_ONLY>(
            arg_idx);

    PacketHeaderPool::reset();
    volatile PACKET_HEADER_TYPE* hdr_fwd = PacketHeaderPool::allocate_header();
    volatile PACKET_HEADER_TYPE* hdr_bwd = PacketHeaderPool::allocate_header();
    hdr_fwd->to_chip_multicast(
        tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_chips - 1 - chip)});
    hdr_bwd->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(chip)});
    if (fabric.is_logically_connected()) {
        fabric.open_finish();
    }

    volatile tt_l1_ptr uint32_t* gather = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_gather));
    volatile tt_l1_ptr uint32_t* ccl_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ccl_sem_addr);
    volatile tt_l1_ptr uint32_t* flag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flag_addr);
    const uint64_t ccl_sem_noc = get_noc_addr(my_x[0], my_y[0], ccl_sem_addr);
    const uint64_t mc_slots = get_noc_multicast_addr(mc_x0, mc_y0, mc_x1, mc_y1, streamer_slots);
    const uint64_t mc_sem = get_noc_multicast_addr(mc_x0, mc_y0, mc_x1, mc_y1, get_semaphore(sem_slots));

    uint32_t expected = 0;
    for (uint32_t l = 0; l < layers; l++) {
        const uint32_t base = slots + (l & 1) * num_chips * vector_bytes;
        noc_semaphore_wait_min(gather, num_streamers * (l + 1));
        if constexpr (num_chips > 1) {
            const uint32_t own = base + chip * vector_bytes;
            size_t rd = own;
            for (uint32_t p = 0; p < packets_per_vector; p++) {
                const uint32_t off = p * packet_bytes;
                const uint32_t bytes = off + packet_bytes <= vector_bytes ? packet_bytes : vector_bytes - off;
                fused_write_atomic_and_advance_local_read_address_for_fabric_write(
                    get_noc_addr(my_x[0], my_y[0], own + off),
                    hdr_fwd,
                    hdr_bwd,
                    fabric,
                    rd,
                    bytes,
                    ccl_sem_noc,
                    1,
                    true);
            }
            expected += (num_chips - 1) * packets_per_vector;
            noc_semaphore_wait_min(ccl_sem, expected);
        }
        if (l + 1 < layers || mcast_last) {
            noc_async_write_multicast_loopback_src(base, mc_slots, num_chips * vector_bytes, mc_dests);
            noc_async_write_barrier();
            *flag = l + 1;
            noc_semaphore_set_multicast_loopback_src(flag_addr, mc_sem, mc_dests);
            noc_async_write_barrier();
        }
    }
    noc_async_write_barrier();
    noc_semaphore_set(ccl_sem, 0);
    if (fabric.is_logically_connected()) {
        fabric.close();
    }
}
