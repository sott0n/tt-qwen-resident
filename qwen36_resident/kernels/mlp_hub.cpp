// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Resident MLP hub (BRISC, one core per chip): pure data movement between the streamers and the fabric.
// Per layer l (slot parity p = l & 1):
//   1. wait until every streamer has written its down columns into slot[p][chip] (gather semaphore),
//   2. line-multicast slot[p][chip] over fabric into slot[p][chip] of every peer hub (fused increment
//      of the peers' CCL semaphore of parity p), wait for the peers' slots. A peer that already has
//      layer l's slots runs ahead and sends layer l + 1's while a slower peer's layer l packets are still
//      in flight, so the two parities count on separate semaphores (a peer cannot get two layers ahead:
//      layer l + 2 needs this chip's layer l + 1 slot, sent after its layer l wait),
//   3. multicast slot[p][0 .. num_chips) to the streamers' partial slots and bump their slots semaphore,
//      one multicast per rectangle of a cover of exactly the streamer cores: the slots are allocated on the
//      streamers only, so any other core under a multicast could hold a CB at that address.
// The last layer's slots stay on the hub for the host to read back (and are also multicast when
// mcast_last is set, for a stage that runs after the last layer).
//
// Compile-time args: 0 num_chips, 1 chip, 2 vector_bytes, 3 packet_bytes, 4 layers, 5 num_streamers,
//   6 sem_gather id, 7 sem_slots id, 8 sem_flag id (local word holding the value multicast into the
//   streamers' slots semaphore), 9 mcast_last, 10 header CB (holds the packet headers when they exceed the
//   per-RISC header pool), 11 published, 12 sem_addr
// Runtime args: 0 slots_addr, 1, 2 ccl semaphore addresses of parity 0, 1 (global semaphores), 3 streamer
// slots addr, 4 rectangles, then per rectangle noc x0, y0, x1, y1, cores, then FabricConnectionManager
// args (fwd, bwd).
// published: the slots are this core's CB 0 (the same address on every chip); a streamer sends the
// streamers' x address into sem_addr and the hub multicasts its slots address into the streamers'
// sem_addr. The hub's compute sums each round's slots into CB 2 and only x is multicast. Arg 0 is then a
// DRAM buffer (0 = none) that gets the last layer's x, arg 3 is unused.

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
constexpr bool mcast_last = get_compile_time_arg_val(9) != 0;
constexpr uint32_t sem_flag = get_compile_time_arg_val(8);
constexpr uint32_t cb_headers = get_compile_time_arg_val(10);
constexpr bool published = get_compile_time_arg_val(11) != 0;
constexpr uint32_t sem_addr = get_compile_time_arg_val(12);
constexpr uint32_t packets_per_vector = (vector_bytes + packet_bytes - 1) / packet_bytes;
constexpr uint32_t num_headers = 2 * packets_per_vector * 2;
constexpr bool pooled = num_headers <= NUM_PACKET_HEADERS / MaxDMProcessorsPerCoreType;

void kernel_main() {
    size_t arg_idx = 0;
    const uint32_t arg0 = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t ccl_sem_addr[2] = {get_arg_val<uint32_t>(arg_idx), get_arg_val<uint32_t>(arg_idx + 1)};
    arg_idx += 2;
    const uint32_t arg3 = get_arg_val<uint32_t>(arg_idx++);
    auto cb_base = [](uint32_t cb) {
        auto& iface = get_local_cb_interface(cb);
        return iface.fifo_limit - iface.fifo_size;
    };
    const uint32_t slots = published ? cb_base(0) : arg0;
    const uint32_t rects = get_arg_val<uint32_t>(arg_idx++);
    const size_t rect_args = arg_idx;
    arg_idx += 5 * rects;
    const uint32_t flag_addr = get_semaphore(sem_flag);
    auto fabric =
        FabricConnectionManager::build_from_args<FabricConnectionManager::BUILD_AND_OPEN_CONNECTION_START_ONLY>(
            arg_idx);

    // one header per (parity, packet, direction), built once: the header writes to the routers are
    // non-blocking, so a header is never rewritten while a send of it may still be reading it
    PacketHeaderPool::reset();
    uint32_t next_hdr = cb_base(cb_headers);
    auto allocate = [&]() -> volatile PACKET_HEADER_TYPE* {
        if constexpr (pooled) {
            return PacketHeaderPool::allocate_header();
        }
        auto* h = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(next_hdr);
        next_hdr += (sizeof(PACKET_HEADER_TYPE) + 15) & ~15u;
        return h;
    };
    volatile PACKET_HEADER_TYPE* hdr[2][packets_per_vector][2];
    for (uint32_t par = 0; par < 2; par++) {
        for (uint32_t p = 0; p < packets_per_vector; p++) {
            const uint32_t off = p * packet_bytes;
            const uint32_t bytes = off + packet_bytes <= vector_bytes ? packet_bytes : vector_bytes - off;
            const uint64_t dst = get_noc_addr(my_x[0], my_y[0], slots + (par * num_chips + chip) * vector_bytes + off);
            const uint64_t sem = get_noc_addr(my_x[0], my_y[0], ccl_sem_addr[par]);
            for (uint32_t dir = 0; dir < 2; dir++) {
                volatile PACKET_HEADER_TYPE* h = allocate();
                h->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{
                    1, static_cast<uint8_t>(dir == 0 ? num_chips - 1 - chip : chip)});
                h->to_noc_fused_unicast_write_atomic_inc(
                    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst, sem, 1, true}, bytes);
                hdr[par][p][dir] = h;
            }
        }
    }
    if (fabric.is_logically_connected()) {
        fabric.open_finish();
    }

    volatile tt_l1_ptr uint32_t* gather = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_gather));
    volatile tt_l1_ptr uint32_t* ccl_sem[2] = {
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ccl_sem_addr[0]),
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ccl_sem_addr[1])};
    volatile tt_l1_ptr uint32_t* flag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flag_addr);
    auto rect = [&](uint32_t r, uint32_t i) { return get_arg_val<uint32_t>(rect_args + 5 * r + i); };

    uint32_t streamer_slots = arg3;
    if constexpr (published) {
        // our slots address to the streamers (from a word past the headers), theirs from a streamer
        volatile tt_l1_ptr uint32_t* word = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(next_hdr);
        *word = slots;
        for (uint32_t r = 0; r < rects; r++) {
            noc_semaphore_set_multicast(
                next_hdr,
                get_noc_multicast_addr(rect(r, 0), rect(r, 1), rect(r, 2), rect(r, 3), get_semaphore(sem_addr)),
                rect(r, 4));
        }
        volatile tt_l1_ptr uint32_t* addr_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_addr));
        while (*addr_sem == 0) {
            invalidate_l1_cache();
        }
        streamer_slots = *addr_sem;
        noc_async_write_barrier();
    }

    uint32_t expected[2] = {0, 0};
    for (uint32_t l = 0; l < layers; l++) {
        const uint32_t par = l & 1;
        const uint32_t base = slots + par * num_chips * vector_bytes;
        noc_semaphore_wait_min(gather, num_streamers * (l + 1));
        if constexpr (num_chips > 1) {
            for (uint32_t p = 0; p < packets_per_vector; p++) {
                const uint32_t src = base + chip * vector_bytes + p * packet_bytes;
                const uint32_t bytes =
                    (p + 1) * packet_bytes <= vector_bytes ? packet_bytes : vector_bytes - p * packet_bytes;
                if (fabric.has_forward_connection()) {
                    perform_payload_send(fabric.get_forward_connection(), src, bytes, hdr[par][p][0]);
                }
                if (fabric.has_backward_connection()) {
                    perform_payload_send(fabric.get_backward_connection(), src, bytes, hdr[par][p][1]);
                }
            }
            expected[par] += (num_chips - 1) * packets_per_vector;
            noc_semaphore_wait_min(ccl_sem[par], expected[par]);
        }
        // published: the compute sums the slots into x
        uint32_t src = base, bytes = num_chips * vector_bytes;
        if constexpr (published) {
            constexpr uint32_t views = vector_bytes / 2048;
            cb_reserve_back(0, num_chips * views);
            cb_push_back(0, num_chips * views);
            cb_wait_front(2, views);
            src = get_read_ptr(2);
            bytes = vector_bytes;
            if (l + 1 == layers && arg0 != 0) {
                const InterleavedAddrGen<true> dump{.bank_base_address = arg0, .page_size = vector_bytes};
                noc_async_write(src, dump.get_noc_addr(0), vector_bytes);
            }
        }
        if (l + 1 < layers || mcast_last) {
            for (uint32_t r = 0; r < rects; r++) {
                noc_async_write_multicast(
                    src,
                    get_noc_multicast_addr(rect(r, 0), rect(r, 1), rect(r, 2), rect(r, 3), streamer_slots),
                    bytes,
                    rect(r, 4));
            }
            noc_async_write_barrier();
            *flag = l + 1;
            for (uint32_t r = 0; r < rects; r++) {
                noc_semaphore_set_multicast(
                    flag_addr,
                    get_noc_multicast_addr(rect(r, 0), rect(r, 1), rect(r, 2), rect(r, 3), get_semaphore(sem_slots)),
                    rect(r, 4));
            }
            noc_async_write_barrier();
        }
        if constexpr (published) {
            noc_async_write_barrier();
            cb_pop_front(2, vector_bytes / 2048);
        }
    }
    noc_async_write_barrier();
    noc_semaphore_set(ccl_sem[0], 0);
    noc_semaphore_set(ccl_sem[1], 0);
    if (fabric.is_logically_connected()) {
        fabric.close();
    }
}
