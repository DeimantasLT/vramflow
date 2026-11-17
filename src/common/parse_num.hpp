// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Griežtas dešimtainių skaičių skaitymas iš teksto (planas docs/planai/2026-11-17-production-ready.md §2.4; clang-tidy
// cert-err34-c). atoi ir sscanf perpildymo nepraneša (sscanf atveju tai neapibrėžta elgsena) ir tyliai priima tarpus bei
// „+“; čia skaičius arba perskaitomas tiksliai, arba atmetamas. Testai — tests/unit/test_parse_num.cpp.
#include <charconv>
#include <cstdint>
#include <map>
#include <string_view>
#include <system_error>

namespace mxlv {

// Dešimtainis sveikasis skaičius s pradžioje: tik skaitmenys (ženkliniam tipui — ir „-“ priekyje), be tarpų ir be „+“.
// Sėkmė — out gauna reikšmę, o s sutrumpėja per perskaitytus simbolius. Ne skaičius arba netelpa į T — false, s ir out
// nepakinta.
template <class T>
bool take_num(std::string_view& s, T& out) {
    T v{};
    const auto r = std::from_chars(s.data(), s.data() + s.size(), v);
    if (r.ec != std::errc{}) return false;
    out = v;
    s.remove_prefix(static_cast<std::size_t>(r.ptr - s.data()));
    return true;
}

// Tikslus simbolis s pradžioje: true — s sutrumpėja per jį.
inline bool take_char(std::string_view& s, char c) {
    if (s.empty() || s.front() != c) return false;
    s.remove_prefix(1);
    return true;
}

// Tarpai, tabuliacijos ir eilučių pabaigos s pradžioje praleidžiami.
inline void skip_spaces(std::string_view& s) {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t' || s.front() == '\n' || s.front() == '\r')) s.remove_prefix(1);
}

// `vramflow-func ref-crc` išvestis ({"fields":[[barkodas,crc],…]}) → {barkodas: crc}. Poros forma — „[N,N]“, N ≤ 2^32 − 1;
// tarpai aplink skaičius leidžiami. Kitokios poros praleidžiamos.
inline std::map<std::uint32_t, std::uint32_t> parse_ref_crc(std::string_view text) {
    std::map<std::uint32_t, std::uint32_t> out;
    for (std::size_t p = text.find('['); p != std::string_view::npos; p = text.find('[', p + 1)) {
        std::string_view v = text.substr(p + 1);
        std::uint32_t mk = 0, crc = 0;
        skip_spaces(v);
        if (!take_num(v, mk)) continue;
        skip_spaces(v);
        if (!take_char(v, ',')) continue;
        skip_spaces(v);
        if (!take_num(v, crc)) continue;
        skip_spaces(v);
        if (take_char(v, ']')) out[mk] = crc;
    }
    return out;
}

}  // namespace mxlv
