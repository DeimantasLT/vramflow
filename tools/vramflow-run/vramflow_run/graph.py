# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""Plan → vaikų procesų komandos (OK-2, T2). Visa grandinė viename domene, flow bazė 0:

    ingest (--flow-base 0; video rolė 7, garsas rolė 8)
      → logo --in-role 7                          (išėjimas rolė 2)
        | gfx (gfxgen pace, rolė 5) + composite --in-role 7 --base gen   (išėjimas rolė 6)
      → enc --in-role 2|6 --audio-role 8 --output …  (garsas — tiesiai iš ingest)

Paleidimo vartai (Proc.wait_for): logo ir composite briaunos laiko žymas (urn:x-vramflow:tag:content-delay-ns) skaito
paleidimo metu — jei įėjimo flow'o dar nėra, laiko jį be žymos (content_delay 0), o enc tada video lygina su ingest
garsu 1 s per anksti (visi grain'ai too_late; išmatuota 2026-11-19). Todėl jie paleidžiami tik tada, kai jų įėjimai jau
yra. enc pats laukia įėjimų (attach_inputs, AudioIn::open) ir žymas skaito po prisijungimo — paleidžiamas iškart, kad
CUDA / NVENC paruošimas sutaptų su ingest zondavimu (OK-2 k. 1: ≤ 5 s iki pirmo paketo).

Rolių numeriai — src/func/flows.hpp (Role). Slaptažodžiai — tik nuorodos `--srt-passphrase-from env:…|file:…`; pati
reikšmė lieka vaiko aplinkoje (Proc.env, paveldėta iš prižiūrėtojo) arba faile — niekada argv.
"""
import dataclasses
import math
import os
from typing import Optional

from .config import Plan
from .redact import redact_url, url_scheme

ROLE_LOGO, ROLE_GFX, ROLE_COMP, ROLE_INGEST, ROLE_INGEST_AUDIO = 2, 5, 6, 7, 8
# OK-2.6: šaltinio spraga — rašytojų politika K3 „freeze“ (paskutinis geras grain'as ≤ --hold-ms 1000, tada juoda), ne
# numatytoji „invalid“: INVALID grain'ą enc laiko aukštupio spraga ir video kadro nekoduoja — išėjimo TS sustotų.
# Spragos lieka suskaičiuotos (ingest gap_fields, logo / composite santraukose — žr. task-4 ataskaitą).
ON_LOSS = ["--on-loss", "freeze"]


def flow_id(role: int, n: int) -> str:
    """Kaip mxlv::func::flow_id: 4d584c47-00RR-4000-8000-%012x."""
    return "4d584c47-%04d-4000-8000-%012x" % (role, n)


@dataclasses.dataclass(frozen=True)
class Proc:
    name: str          # vaiko vardas žurnaluose ir --out kataloge (ingest, logo, gfx, composite, enc)
    argv: list
    env: dict
    flows: tuple = ()     # flow'ai, kuriuos vaikas sukuria: visi atsirado → vaikas pasiruošęs
    wait_for: tuple = ()  # paleidimo vartai: flow'ai, kurie turi būti domene prieš paleidžiant (žr. commands)


def _num(v) -> str:
    return f"{v:g}" if isinstance(v, float) else str(v)


def muxrate(vbitrate: int, abitrate: int) -> int:
    """CBR TS sparta: (video + garsas) · 1,1, į viršų iki 100 kbit/s (enc numatytosios 7,5 M + 192 k → 8,5 M)."""
    return math.ceil((vbitrate + abitrate) * 11 / 10 / 100_000) * 100_000


def _secret(items):
    refs = [it.srt_passphrase_from for it in items if url_scheme(it.url) == "srt" and it.srt_passphrase_from]
    return ["--srt-passphrase-from", refs[0]] if refs else []


def commands(plan: Plan, domain: str, exe: str, out_dir: Optional[str] = None, base_env: Optional[dict] = None) -> list:
    """Vaikų komandos paleidimo tvarka. out_dir — kiekvienam vaikui --out <out_dir>/<vardas> ir --stats-file …/stats.jsonl."""
    N = len(plan.inputs)
    env = dict(os.environ if base_env is None else base_env)
    procs = []

    def add(name, sub, args, flows=(), wait_for=()):
        argv = [exe, sub, "--domain", str(domain), "--flows", str(N), "--storage", plan.storage, "--format", plan.format]
        if out_dir is not None:
            d = f"{out_dir}/{name}"
            argv += ["--out", d, "--stats-file", f"{d}/stats.jsonl"]
        procs.append(Proc(name, argv + [str(a) for a in args], env, tuple(flows), tuple(wait_for)))

    ins = [a for it in plan.inputs for a in ("--input", it.url)]
    add("ingest", "ingest", ["--flow-base", 0, *ins, *ON_LOSS, "--input-wait-s", _num(plan.input_wait_s), *_secret(plan.inputs)],
        [flow_id(r, n) for n in range(N) for r in (ROLE_INGEST, ROLE_INGEST_AUDIO)])
    base = ["--in-role", ROLE_INGEST, "--in-flow-base", 0, "--flow-base", 0]
    video = [flow_id(ROLE_INGEST, n) for n in range(N)]
    if plan.process.kind == "logo":
        lg = plan.process.logo
        xy = (["--logo-x", lg.x] if lg.x is not None else []) + (["--logo-y", lg.y] if lg.y is not None else [])
        add("logo", "logo", [*base, *ON_LOSS, "--logo", lg.file, *xy], [flow_id(ROLE_LOGO, n) for n in range(N)], video)
        enc_in = ROLE_LOGO
    else:
        layers = plan.process.composite.layers
        K = len(layers)
        gfx = [flow_id(ROLE_GFX, n * 8 + k) for n in range(N) for k in range(K)]
        add("gfx", "gfxgen", ["--flow-base", 0, "--layers", K, "--layer-geom", ",".join(L.geom for L in layers), "--mode", "pace"], gfx)
        xy = [a for L in layers for a in ("--layer-xy", f"{L.x},{L.y}")]
        op = [a for L in layers for a in ("--opacity", _num(L.opacity))]
        add("composite", "composite", [*base, *ON_LOSS, "--base", "gen", "--layers", K, *xy, *op, "--out-format", "v210"],
            [flow_id(ROLE_COMP, n) for n in range(N)], video + gfx)
        enc_in = ROLE_COMP
    o0 = plan.outputs[0]  # config: visų išėjimų spartos vienodos
    outs = [a for it in plan.outputs for a in ("--output", it.url)]
    add("enc", "enc", ["--in-role", enc_in, "--in-flow-base", 0, "--flow-base", 0, "--audio", 1, "--audio-role", ROLE_INGEST_AUDIO,
                       "--encoder", "nvenc", "--vbitrate", o0.vbitrate, "--abitrate", o0.abitrate,
                       "--muxrate", muxrate(o0.vbitrate, o0.abitrate), *outs, *_secret(plan.outputs)])
    return procs


def display(p: Proc) -> str:
    """Komanda žurnalui: URL be paslapčių (redact_url)."""
    return " ".join(redact_url(a) if url_scheme(a) and "://" in a else a for a in p.argv)
