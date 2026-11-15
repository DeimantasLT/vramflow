// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// Paketų eilė (round 10, M1): riba elementais ir baitais, perpildymas skaičiuojamas (išmesti paketai, episodai).
#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cstdint>

#include "ingest/packet_queue.hpp"

using namespace mxlv::ingest;

namespace {

struct Item {
    std::uint32_t size = 0;  // 0 — žymė (sesija, flush, no_stream)
    int id = 0;
    int* freed = nullptr;
};

struct ItemTraits {
    static std::size_t bytes(const Item& i) { return i.size; }
    static void drop(Item& i) {
        if (i.freed) ++*i.freed;
    }
};

using Q = PacketQueue<Item, ItemTraits>;

int pop_id(Q& q) {
    Item it;
    REQUIRE(q.pop(it, std::chrono::milliseconds(0)));
    return it.id;
}

}  // namespace

TEST_CASE("packet_queue: riba baitais — seniausi paketai išmetami ir skaičiuojami (M1)", "[ingest][queue]") {
    int freed = 0;
    Q q(1000, 10'000);
    for (int k = 0; k < 10; ++k) q.push(Item{1000, k, &freed});
    CHECK(q.bytes() == 10'000);
    CHECK(q.dropped() == 0);
    q.push(Item{2500, 10, &freed});  // reikia 2500 B vietos — išmetami 3 seniausi
    CHECK(q.dropped() == 3);
    CHECK(freed == 3);
    CHECK(q.bytes() == 9'500);
    CHECK(q.overflow_episodes() == 1);
    CHECK(pop_id(q) == 3);
    CHECK(q.bytes() == 8'500);
    // nauja perpildymo serija — antras episodas (žurnalui: viena eilutė seriai, ne paketui)
    q.push(Item{100, 11, &freed});  // telpa
    q.push(Item{5000, 12, &freed});
    CHECK(q.overflow_episodes() == 2);
    CHECK(q.bytes() <= 10'000);
}

TEST_CASE("packet_queue: riba elementais lieka; baitų riba keičiama", "[ingest][queue]") {
    int freed = 0;
    Q q(4, 1u << 30);
    for (int k = 0; k < 6; ++k) q.push(Item{10, k, &freed});
    CHECK(q.size() == 4);
    CHECK(q.dropped() == 2);
    CHECK(pop_id(q) == 2);
    q.set_cap_bytes(25);
    q.push(Item{10, 6, &freed});
    CHECK(q.bytes() <= 25);
    q.clear();
    CHECK(q.bytes() == 0);
    CHECK(q.size() == 0);
}

TEST_CASE("packet_queue: drop_packets palieka žymes, baitai — 0", "[ingest][queue]") {
    int freed = 0;
    Q q(100, 1u << 20);
    q.push(Item{10, 1, &freed});
    q.push(Item{0, 2, &freed});
    q.push(Item{10, 3, &freed});
    CHECK(q.drop_packets() == 2);
    CHECK(q.bytes() == 0);
    CHECK(q.size() == 1);
    CHECK(pop_id(q) == 2);
}
