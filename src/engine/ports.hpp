// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Variklio įėjimų / išėjimų sąsajos be MXL tipų (planas §7 ports.hpp). MXL įgyvendinimai — mxlv_engine_mxl (V7);
// testuose — ScriptedInput / RecordingOutput (tests/engine/fakes.hpp). Visus metodus kviečia tik variklio gija.
#include <cstdint>
#include <string>

#include "engine/clock.hpp"
#include "engine/gap_state.hpp"
#include "engine/input_state.hpp"

namespace mxlv::engine {

enum class Probe : std::uint8_t { Ready, Invalid, NotYet, TooLate, Gone, Error };
enum class Need : std::uint8_t { Required, Wanted, Optional };  // Wanted: laukiama, bet vienetas tęsia be jo (enc garsas)
enum class InMode : std::uint8_t { Exact, Latest };             // Latest: naujausias ne vėlesnis nei j (mv plytelė)

struct GrainRef {
    const void* ptr = nullptr;
    std::uint32_t flags = 0;
};
// Briaunos sutartis (§6 D7, V24; engine/edge.hpp): expect_ns — aukštupio skelbiama commit'o riba po T(j) (žyma
// commit-bound-ns arba --input-expect-ms); variklio fazė E = max(expect_ns) per visus įėjimus. content_delay_ns — aukštupio
// turinio vėlinimas (žyma content-delay-ns), išėjimo content_delay skaičiuoti.
struct EdgeTiming {
    std::int32_t index_offset = 0;  // ≤ 0: įėjimo indeksas j = i + index_offset
    std::int64_t expect_ns = 0;
    std::int64_t content_delay_ns = 0;
};

// Flow adresas (domenas, flow UUID). Planas nurodo runtime::FlowRef iš [P] E0 — jo dar nėra, todėl kol kas čia.
struct FlowRef {
    std::string domain;
    std::string flow_id;
};

class InputPort {
public:
    virtual ~InputPort() = default;
    virtual bool attach_poll(std::uint64_t now_tai) = 0;  // neblokuoja; true — prijungtas
    virtual bool mismatch(std::string& why) const = 0;
    virtual Index head() = 0;                             // be sisteminio kvietimo; kNoIndex — nežinoma
    virtual std::uint64_t last_write_tai() = 0;           // ašies parinkimui
    virtual Probe probe(Index j, GrainRef& out) = 0;      // neblokuoja
    virtual Probe wait(Index j, std::uint64_t timeout_ns, GrainRef& out) = 0;  // blokuoja ≤ timeout
    virtual bool writer_alive() = 0;                      // tik ties A riba ir kas 100 ms Stalled būsenoje
    virtual bool lapped(Index j, Index now_idx) = 0;
    virtual void detach() = 0;
    virtual void retarget(const FlowRef&) = 0;
};

enum class OpenResult : std::uint8_t { Ok, AlreadyCommitted, Error };  // AlreadyCommitted — laikrodis atgal (M11)

class OutputPort {
public:
    virtual ~OutputPort() = default;
    virtual OpenResult open(Index i, void*& slot) = 0;
    virtual bool commit() = 0;  // validSlices = totalSlices
    virtual void cancel() = 0;
    virtual bool fill_gap(Index i, GapAction a, Index repeat_from) = 0;  // kopijos — į funkcijos stream'ą
    virtual bool flush_gaps() = 0;                                       // vienas sync fazei (host — nieko)
    virtual bool interlaced() const = 0;
    virtual std::uint32_t slot_count() const = 0;
    // Sugedusio vieneto (K17, K-T11) išėjimas atlaisvinamas: žemupys mato flow dingusį (Gone / Detached), ne amžiną INVALID.
    // Po release() variklis prievado nebekviečia.
    virtual void release() {}
    // Sugedusio vieneto atkūrimas (D8: bandoma su atidėjimu 1, 2, 4 … ≤ 30 s): išėjimas sukuriamas iš naujo. Kviečia variklio
    // atkūrimo gija (ne tick'as), kol variklis šio prievado nekviečia. true — vėl veikia (variklis tęsia); false — nepavyko.
    virtual bool reopen() { return false; }
};

}  // namespace mxlv::engine
