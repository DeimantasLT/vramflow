// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Dekoderio tiekimo vartai (planas docs/planai/2026-11-11-production-grade.md §P3; brief'o pamokos 3 ir 4). Gryna logika —
// be libav, MXL ir GPU (tikrinama tests/unit/test_feed_gate.cpp); P4 ją plečia priėmimo kontrole.
//
//   Atsarga (pamoka 3): paketas į dekoderį leidžiamas ne anksčiau kaip lead_ns prieš jo DTS laiką TAI — dekoderiui (NVDEC per
//   h264_cuvid išveda kadrą 4–8 kadrais vėliau) lieka laiko, o laikomų kadrų (VRAM) skaičius ribotas ≈ lead · kadrų dažnis.
//   Paketai laukia suspausti (host RAM — pigu).
//   Perkrova (pamoka 4): paketas, kurio kadras nebespės savo laiko (PTS TAI − dabar < min_ns), nededamas į dekoderį — vėluojantis
//   dekoderis kitaip dekoduotų vien pavėluotus kadrus ir niekada nepasivytų. Nuo tokio paketo metama iki kito rakto kadro, kuris
//   dar spėja (dekoderis išvalomas — be atramos kadrų likusieji tik iškraipytų vaizdą); tada tęsiama.
#include <cstdint>

namespace mxlv::ingest {

struct FeedGateConfig {
    std::int64_t lead_ns = 400'000'000;  // --decode-lead-ms
    std::int64_t min_ns = 0;             // kadras, kurio PTS TAI − dabar < min_ns, nebespės (0 — tik jau praėję)
};

enum class Feed : std::uint8_t { Hold, Send };

// Send, kai dts_tai − now ≤ lead (ir kai DTS jau praeityje); kitaip Hold.
constexpr Feed feed_decision(std::int64_t dts_tai_ns, std::int64_t now_tai_ns, const FeedGateConfig& c) {
    return dts_tai_ns - now_tai_ns <= c.lead_ns ? Feed::Send : Feed::Hold;
}

// Perkrovos vartai: paketų seka po vieną (DTS tvarka). Grąžina true — paketą siųsti į dekoderį; false — mesti. flush — prieš
// šį (rakto) paketą išvalyti dekoderį (pirmas paketas po praleidimo). Praleidimų skaičius — skip_runs, išmesti paketai — dropped.
class SkipToKey {
public:
    bool admit(std::int64_t pts_tai_ns, bool key, std::int64_t now_tai_ns, const FeedGateConfig& c, bool& flush) {
        flush = false;
        const bool late = pts_tai_ns - now_tai_ns < c.min_ns;
        if (skipping_) {
            if (!key || late) {
                ++dropped_;
                return false;
            }
            skipping_ = false;
            flush = true;  // naujas GOP: senos atramos nebereikalingos
            return true;
        }
        if (!late) return true;
        skipping_ = true;
        ++skip_runs_;
        ++dropped_;
        return false;
    }
    void reset() { skipping_ = false; }
    bool skipping() const { return skipping_; }
    std::uint64_t skip_runs() const { return skip_runs_; }
    std::uint64_t dropped() const { return dropped_; }

private:
    bool skipping_ = false;
    std::uint64_t skip_runs_ = 0, dropped_ = 0;
};

}  // namespace mxlv::ingest
