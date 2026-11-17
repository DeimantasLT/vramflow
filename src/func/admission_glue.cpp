// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// Priėmimo kontrolė funkcijoms (planas docs/planai/2026-11-11-production-grade.md §P4): talpos failas, knyga, srautų priėmimas
// iš eilės. Ką tikrina kiekviena funkcija (docs/rezultatai/2026-11-17-p4-admission.md):
//   ingest, dec — NVDEC Mpix/s + VRAM (išmatuotas functions.<žymė> arba įvertis runtime::ingest_demand);
//   enc nvenc   — NVENC Mpix/s + NVENC sesijos + VRAM (tik išmatuotas functions.enc-nvenc-<tune>);
//   enc x264, gen, gfxgen, logo, composite — tik VRAM (išmatuotas functions.<žymė>; nėra — netikrinama, eilutė stderr);
//   verify ir --impl cpu — netaikoma (testo skaitytuvas; GPU nenaudojama).
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "func/engine_glue.hpp"
#include "func/func_common.hpp"
#include "func/stats_lines.hpp"
#include "mxlio/cuda_uuid.hpp"

namespace mxlv::func {

namespace {
std::string mb(std::uint64_t b) {
    char s[32];
    std::snprintf(s, sizeof s, "%.0f", static_cast<double>(b) / (1024.0 * 1024.0));
    return s;
}
std::string num2(double v) {  // lietuviškas dešimtainis kablelis, kaip runtime::describe
    char s[32];
    std::snprintf(s, sizeof s, "%.2f", v);
    for (char* c = s; *c; ++c)
        if (*c == '.') *c = ',';
    return s;
}
const char* mode_name(runtime::AdmissionMode m) {
    return m == runtime::AdmissionMode::Enforce ? "enforce" : m == runtime::AdmissionMode::Warn ? "warn" : "off";
}
// Ingest / dec: išėjimo flow'o ring'o grain'ų skaičius (domeno istorija × grain'ų dažnis + 1)
std::uint32_t ring_grains(const std::string& domain, const vramflow::VideoFormat& f) {
    const vramflow::Rational g = f.grain_rate();
    const double n = static_cast<double>(domain_history_ns(domain)) / 1e9 * static_cast<double>(g.num) / static_cast<double>(g.den);
    return static_cast<std::uint32_t>(std::ceil(n)) + 1;
}
}  // namespace

AdmissionRun& admission() {
    static AdmissionRun a;
    return a;
}

int admission_begin(const Opts& o, const std::string& role, std::string& fatal) {
    AdmissionRun& a = admission();
    a.role = role;
    if (!runtime::parse_admission_mode(o.admission, a.mode)) {
        fatal = "--admission: " + o.admission + " (off | warn | enforce)";
        return 1;
    }
    if (a.mode == runtime::AdmissionMode::Off || o.impl == "cpu" || role == "verify") return 0;
    const int dev = cuda_device(o.device_uuid);
    if (dev < 0) return 0;  // funkcija pati praneš „CUDA įrenginys“ (exit 2)
    a.active = true;
    const std::string uuid = mxlv::cuda_uuid_string(dev);
    cudaDeviceProp p{};
    std::uint64_t total = 0;
    if (cudaGetDeviceProperties(&p, dev) == cudaSuccess) total = p.totalGlobalMem;
    else (void)cudaGetLastError();
    const char* env = std::getenv("VRAMFLOW_CAPACITY");
    const std::string path = !o.capacity_file.empty() ? o.capacity_file : env ? std::string(env) : std::string{};
    std::string err;
    a.cap.headroom = o.admission_headroom;
    bool ok = false;
    if (path.empty())
        err = "talpa neišmatuota: nenurodytas --capacity-file (arba VRAMFLOW_CAPACITY)";
    else
        ok = runtime::load_capacity(path, uuid, a.cap, err);
    if (ok && !runtime::Ledger::open(o.domain, uuid, a.ledger, err)) {
        if (a.mode == runtime::AdmissionMode::Enforce) {
            fatal = "priėmimo kontrolė: " + err;
            return 2;
        }
        ok = false;
    }
    if (!ok) {
        if (a.mode == runtime::AdmissionMode::Enforce) {
            fatal = "priėmimo kontrolė (--admission enforce): talpa neišmatuota, paleiskite 70-capacity (scripts/linux/70-capacity.sh, "
                    "tada tests/perf/capacity_file.py) ir nurodykite --capacity-file — " + err;
            return 1;
        }
        std::fprintf(stderr, "[%s] įspėjimas: priėmimo kontrolė (warn) — talpa neišmatuota, srautai netikrinami (%s; "
                     "matavimas: scripts/linux/70-capacity.sh + tests/perf/capacity_file.py)\n", role.c_str(), err.c_str());
        return 0;
    }
    if (a.cap.vram_bytes == 0) a.cap.vram_bytes = total;
    a.have_capacity = true;
    for (const std::string& w : runtime::capacity_warnings(a.cap, nvml_driver_version()))
        std::fprintf(stderr, "[%s] įspėjimas: priėmimo kontrolė — %s\n", role.c_str(), w.c_str());
    std::fprintf(stderr, "[%s] priėmimo kontrolė (%s): talpa %s — nvdec %s, nvenc %s Mpix/s, nvenc sesijos %d, VRAM %s MB, atsarga %s\n",
                 role.c_str(), mode_name(a.mode), a.cap.source.c_str(), num2(a.cap.nvdec_mpix_s).c_str(), num2(a.cap.nvenc_mpix_s).c_str(),
                 a.cap.nvenc_sessions, mb(a.cap.vram_bytes).c_str(), num2(a.cap.headroom).c_str());
    return 0;
}

std::size_t admission_admit(const std::string& tag, std::vector<runtime::Demand> flows, const std::vector<std::string>& names) {
    AdmissionRun& a = admission();
    a.requested += flows.size();
    if (!a.active || !a.have_capacity) {
        a.admitted += flows.size();
        return flows.size();
    }
    runtime::Demand proc;
    const auto it = a.cap.functions.find(tag);
    if (it != a.cap.functions.end() && it->second.vram_per_flow_bytes > 0) {
        // Išmatuotas VRAM modelis (P8): proceso VRAM = offset + N · per_flow
        proc.vram_bytes = it->second.vram_offset_bytes;
        for (runtime::Demand& d : flows) d.vram_bytes = it->second.vram_per_flow_bytes;
    } else {
        bool any_vram = false;
        for (const runtime::Demand& d : flows) any_vram = any_vram || d.vram_bytes > 0;
        std::fprintf(stderr, "[%s] priėmimo kontrolė: talpos faile nėra functions.%s.vram_per_flow_bytes — VRAM %s\n", a.role.c_str(),
                     tag.c_str(), any_vram ? "tikrinama pagal įvertį" : "netikrinama");
    }
    const runtime::AdmitResult r = runtime::admit_in_order(a.ledger, a.role + "-" + tag, proc, flows, a.cap, a.mode);
    for (std::size_t k = 0; k < flows.size(); ++k) {
        const runtime::Decision& d = r.decisions[k];
        if (d.admit) continue;
        const std::string who = k < names.size() ? names[k] : "srautas " + std::to_string(k);
        std::string line;
        if (k < r.admitted)  // warn: priimtas viršijus talpą
            line = who + ": viršija talpą (paleidžiamas, --admission warn) — " + runtime::describe(d);
        else if (d.engine == "order")
            line = who + ": atmestas — ankstesnis srautas netilpo (priimama tik iš eilės)";
        else
            line = who + ": atmestas — " + runtime::describe(d) + (r.process_refused ? " (proceso pastovi dalis)" : "");
        std::fprintf(stderr, "[%s] priėmimo kontrolė: %s (talpa %s)\n", a.role.c_str(), line.c_str(), a.cap.source.c_str());
        a.notes.push_back(line);
    }
    a.admitted += r.admitted;
    a.refused += flows.size() - r.admitted;
    admission_refused() = a.refused;
    a.over += r.over;
    const runtime::Demand t = a.ledger.total();
    std::fprintf(stderr, "[%s] priėmimo kontrolė (%s): priimta %zu iš %zu; užimta GPU: nvdec %s, nvenc %s Mpix/s, nvenc sesijos %d, VRAM %s MB\n",
                 a.role.c_str(), mode_name(a.mode), r.admitted, flows.size(), num2(t.nvdec_mpix_s).c_str(), num2(t.nvenc_mpix_s).c_str(),
                 t.nvenc_sessions, mb(t.vram_bytes).c_str());
    return r.admitted;
}

std::vector<runtime::Demand> admission_demand(const Opts& o, const std::string& role, std::string& tag) {
    std::vector<runtime::Demand> v(o.flows);
    tag = role;
    if (role == "dec") {
        // NVDEC (h264_cuvid / nvdec): DPB ≤ 16 + atkūrimo buferis + 2; ring'as — domeno istorija
        const runtime::Demand d = runtime::ingest_demand(o.fmt, 16 + std::max(o.playout_frames, 0) + 2, ring_grains(o.domain, o.fmt), true);
        for (runtime::Demand& x : v) x = d;
    } else if (role == "enc") {
        if (o.encoder == "nvenc") {
            tag = "enc-nvenc-" + o.tune;
            for (runtime::Demand& x : v) {
                x.nvenc_mpix_s = runtime::mpix_s(o.fmt);
                x.nvenc_sessions = 1;
            }
        } else {
            tag = "enc-x264-" + o.x264_preset;
        }
    } else if (role == "composite") {
        tag = "composite-k" + std::to_string(o.layers);
    }
    return v;
}

}  // namespace mxlv::func
