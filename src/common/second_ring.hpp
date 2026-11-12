// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Fiksuoto dydžio žiedas „sekundė → reikšmė“ (paskutinės capacity() sekundžių su duomenimis). Tick'o kelyje: max / add —
// be alokacijų (buferis paruošiamas konstruktoriuje / set_capacity); senesnės sekundės perrašomos. Tarnybos režime (paros
// trukmės paleidimai) atmintis nebeauga: anksčiau std::map kas sekundę kas flow'ui pridėdavo mazgą (~5 MB / parą / flow).
// Sekundės ateina didėjančia tvarka (indekso laikas T(i)); šiek tiek senesnė (pernešimas į kitą tick'ą, praleidimų
// atkarpa) randama tarp paskutinių kSearch įrašų, dar senesnė — atmetama ir suskaičiuojama (dropped()).
#include <cstddef>
#include <cstdint>
#include <vector>

namespace mxlv {

class SecondRing {
public:
    static constexpr std::size_t kDefault = 3600;  // 1 val.
    static constexpr std::size_t kSearch = 8;

    explicit SecondRing(std::size_t capacity = kDefault) { set_capacity(capacity); }
    // Alokuoja (ne tick'e): turinys išvalomas.
    void set_capacity(std::size_t capacity) {
        e_.assign(capacity == 0 ? 1 : capacity, E{});
        head_ = n_ = 0;
        dropped_ = 0;
    }
    void max(std::uint64_t sec, std::uint64_t v) {
        if (E* e = slot(sec)) e->v = e->v > v ? e->v : v;
    }
    void add(std::uint64_t sec, std::uint64_t v = 1) {
        if (E* e = slot(sec)) e->v += v;
    }
    // Seniausia → naujausia: f(sekundė, reikšmė).
    template <class F>
    void for_each(F f) const {
        for (std::size_t k = 0; k < n_; ++k) {
            const E& e = e_[(head_ + e_.size() - n_ + k) % e_.size()];
            f(e.sec, e.v);
        }
    }
    bool empty() const { return n_ == 0; }
    std::size_t size() const { return n_; }
    std::size_t capacity() const { return e_.size(); }
    std::uint64_t dropped() const { return dropped_; }  // per senos sekundės (už paieškos ar žiedo ribų)

private:
    struct E {
        std::uint64_t sec = 0, v = 0;
    };
    E& at_back(std::size_t k) { return e_[(head_ + e_.size() - 1 - k) % e_.size()]; }  // k = 0 — naujausia
    E* slot(std::uint64_t sec) {
        if (n_ != 0) {
            E& last = at_back(0);
            if (sec == last.sec) return &last;
            if (sec < last.sec) {
                for (std::size_t k = 1; k < n_ && k < kSearch; ++k) {
                    E& e = at_back(k);
                    if (e.sec == sec) return &e;
                    if (e.sec < sec) break;  // tokios sekundės žiede nėra (tarp įrašų) — neįterpiama
                }
                ++dropped_;
                return nullptr;
            }
        }
        E& e = e_[head_];
        e = E{sec, 0};
        head_ = (head_ + 1) % e_.size();
        if (n_ < e_.size()) ++n_;
        return &e;
    }

    std::vector<E> e_;
    std::size_t head_ = 0, n_ = 0;  // head_ — kitas rašomas
    std::uint64_t dropped_ = 0;
};

}  // namespace mxlv
