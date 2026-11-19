# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""`vramflow-run` konfigūracija (reikalavimai docs/reikalavimai/2026-11-19-operatoriaus-kelias.md OK-2, T3).

Vienas JSON failas, schema v1, griežta: nežinomas ar pasikartojantis laukas — klaida (anti-tikslas „tyliai ignoruojamas
laukas“). Kiekviena klaida — `ConfigError(path, why)`, `str()` = "inputs[1].url: …". Slaptažodžiai — tik nuorodos
(`srt_passphrase_from`: `env:VARDAS` | `file:/kelias`); pati paslaptis į Plan nepatenka.

Ribos, kurias nulemia vaikų procesai (ne schema):
* vienas `ingest` ir vienas `enc` procesas visiems srautams — todėl visi srt įėjimai (ir visi srt išėjimai) turi tą patį
  `srt_passphrase_from`, o visi išėjimai — tą pačią `vbitrate` / `abitrate`;
* `format` — tik tie, kuriuos priima C++ `parse_format` + `format_supported` (src/common/format.hpp).
"""
import dataclasses
import ipaddress
import json
import math
import os
import pathlib
import re
import struct
from typing import Optional, Union

from .redact import redact_url, url_scheme

MAX_STREAMS = 16
MAX_LAYERS = 8
# src/common/format.hpp: kSizes × kProgRates / kIntRates (interlaced — tik 1080)
_PROG = ("25", "50", "29.97", "59.94")
FORMATS = tuple([f"{h}p{r}" for h in (720, 1080, 2160) for r in _PROG] + [f"1080i{r}" for r in ("50", "59.94")])
RASTER = {"720": (1280, 720), "1080": (1920, 1080), "2160": (3840, 2160)}
STORAGES = ("cuda-linear", "host")
IN_SCHEMES = ("udp", "rtp", "srt", "http", "https")
OUT_SCHEMES = ("udp", "srt")
BUILTIN_LOGO = (384, 216)  # media::ref::builtin_logo()
LAYER_LOGO = (384, 216)    # gfxgen --layer-geom logo
SOURCES = ("synthetic",)   # libvramflow-gfx šaltiniai, kuriuos prižiūrėtojas paleidžia pats (gfxgen pace)
DEFAULT_VBITRATE, DEFAULT_ABITRATE = 7_500_000, 192_000  # kaip enc numatytosios
VBITRATE_RANGE, ABITRATE_RANGE = (500_000, 50_000_000), (32_000, 512_000)
_SECRET_REF = re.compile(r"^(env:[A-Za-z_][A-Za-z0-9_]*|file:.+)$")


class ConfigError(Exception):
    """Konfigūracijos klaida: path — lauko kelias JSON'e ("" — visas failas), why — priežastis."""

    def __init__(self, path: str, why: str):
        super().__init__(f"{path}: {why}" if path else why)
        self.path, self.why = path, why


@dataclasses.dataclass(frozen=True)
class Input:
    url: str
    srt_passphrase_from: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class Output:
    url: str
    vbitrate: int = DEFAULT_VBITRATE
    abitrate: int = DEFAULT_ABITRATE
    srt_passphrase_from: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class Logo:
    file: str = "builtin"   # "builtin" arba PNG kelias
    x: Optional[int] = None  # None — logo numatytoji vieta (viršutinis dešinys kampas)
    y: Optional[int] = None


@dataclasses.dataclass(frozen=True)
class Layer:
    source: str
    geom: str
    x: int
    y: int
    opacity: float


@dataclasses.dataclass(frozen=True)
class Composite:
    layers: tuple


@dataclasses.dataclass(frozen=True)
class Process:
    kind: str  # "logo" | "composite"
    logo: Optional[Logo] = None
    composite: Optional[Composite] = None


@dataclasses.dataclass(frozen=True)
class Restart:
    max: int = 5
    window_s: Union[int, float] = 60


@dataclasses.dataclass(frozen=True)
class Plan:
    version: int
    format: str
    storage: str
    inputs: tuple
    process: Process
    outputs: tuple
    restart: Restart
    input_wait_s: Union[int, float]
    source: str = ""  # konfigūracijos failo kelias (pranešimams)

    @property
    def raster(self):
        return RASTER[re.match(r"\d+", self.format).group(0)]

    @property
    def interlaced(self):
        return "i" in self.format


# --- JSON su pasikartojančių raktų aptikimu ---------------------------------------------------------------------------

class _Obj(dict):
    """JSON objektas; dups — pasikartojantys raktai (json.loads juos tyliai perrašytų)."""
    dups = ()


def _pairs(pairs):
    o = _Obj()
    seen, dups = set(), []
    for k, v in pairs:
        if k in seen:
            dups.append(k)
        seen.add(k)
        o[k] = v
    o.dups = tuple(dups)
    return o


def _join(path, key):
    return f"{path}.{key}" if path else key


def _obj(v, path, allowed, required=()):
    if not isinstance(v, dict):
        raise ConfigError(path, "turi būti objektas {…}")
    for k in getattr(v, "dups", ()):
        raise ConfigError(_join(path, k), "laukas pasikartoja")
    for k in v:
        if k not in allowed:
            raise ConfigError(_join(path, k), "nežinomas laukas")
    for k in required:
        if k not in v:
            raise ConfigError(_join(path, k), "privalomas laukas")
    return v


def _int(v, path, lo, hi):
    if isinstance(v, bool) or not isinstance(v, int):
        raise ConfigError(path, f"turi būti sveikasis skaičius {lo}…{hi}")
    if not lo <= v <= hi:
        raise ConfigError(path, f"{v} — leidžiama {lo}…{hi}")
    return v


def _num(v, path, lo_excl, hi):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ConfigError(path, f"turi būti skaičius (> {lo_excl}, ≤ {hi})")
    if not lo_excl < v <= hi:
        raise ConfigError(path, f"{v} — leidžiama > {lo_excl} ir ≤ {hi}")
    return v


def _str(v, path):
    if not isinstance(v, str) or not v:
        raise ConfigError(path, "turi būti netuščia eilutė")
    return v


def _list(v, path, lo, hi, what):
    if not isinstance(v, list):
        raise ConfigError(path, "turi būti sąrašas […]")
    if not lo <= len(v) <= hi:
        raise ConfigError(path, f"{len(v)} {what} — leidžiama {lo}…{hi} (≤ {hi})")
    return v


# --- URL ---------------------------------------------------------------------------------------------------------------

def _query_pairs(url):
    q = url.find("?")
    if q < 0:
        return []
    query = url[q + 1:].split("#", 1)[0]
    return [kv.split("=", 1) if "=" in kv else [kv, None] for kv in query.split("&") if kv]


def _endpoint(url):
    """(schema, host, portas) arba None, jei autoritete nėra host:port. IPv6 — [..]:portas."""
    s = url_scheme(url)
    a0 = len(s) + 3
    a1 = min([i for i in (url.find(c, a0) for c in "/?#") if i >= 0] or [len(url)])
    auth = url[a0:a1].rsplit("@", 1)[-1]
    m = re.fullmatch(r"(\[[0-9A-Fa-f:.]+\]|[^:\[\]]+):(\d{1,5})", auth)
    if not m:
        return None
    return s, m.group(1).lower(), int(m.group(2))


def _url(v, path, schemes):
    url = _str(v, path)
    shown = redact_url(url)
    if any(k.lower() == "passphrase" for k, _ in _query_pairs(url)) or "passphrase=" in url.lower():
        raise ConfigError(path, "slaptažodis URL'e neleidžiamas — naudokite srt_passphrase_from (env:VARDAS arba file:/kelias)")
    s = url_scheme(url)
    if s not in schemes:
        raise ConfigError(path, f"protokolas „{s or 'vietinis failas'}“ nepalaikomas (leidžiami: {', '.join(schemes)}): {shown}")
    if url[len(s):len(s) + 3] != "://":
        raise ConfigError(path, f"URL turi prasidėti „{s}://“: {shown}")
    ep = _endpoint(url)
    if s in ("udp", "rtp", "srt") and (ep is None or not 1 <= ep[2] <= 65535):  # http(s) — portas nebūtinas
        raise ConfigError(path, f"adresas be galiojančio host:portas (portas 1–65535): {shown}")
    if s == "srt":
        for k, val in _query_pairs(url):
            if k == "mode" and val not in ("listener", "caller"):
                raise ConfigError(path, f"srt mode — tik listener arba caller: {shown}")
    return url


def _secret_ref(v, path, url):
    if v is None:
        return None
    if not isinstance(v, str) or not _SECRET_REF.match(v):
        raise ConfigError(path, "turi būti env:VARDAS arba file:/kelias (pats slaptažodis čia nerašomas)")
    if url_scheme(url) != "srt":
        raise ConfigError(path, "leidžiama tik srt:// adresui")
    return v


def _same_secret(items, prefix):
    """Vienas procesas (ingest / enc) — viena paslaptis visiems srt adresams."""
    srt = [(k, it) for k, it in enumerate(items) if url_scheme(it.url) == "srt"]
    for k, it in srt[1:]:
        if it.srt_passphrase_from != srt[0][1].srt_passphrase_from:
            raise ConfigError(f"{prefix}[{k}].srt_passphrase_from",
                              f"turi sutapti su {prefix}[{srt[0][0]}].srt_passphrase_from (vienas procesas — vienas slaptažodis "
                              "visiems srt adresams)")


def _is_multicast(host):
    try:
        return ipaddress.ip_address(host.strip("[]")).is_multicast
    except ValueError:
        return False


def _local_ports(url, is_input):
    """Vietiniai UDP portai, prie kurių adresas prisiriša: udp / rtp įėjimai (rtp — ir RTCP portas + 1), srt listener (įėjimas ar
    išėjimas; SRT veikia per UDP). Siunčiantys (udp išėjimas, srt caller) ir http — nesiriša ([])."""
    s, ep = url_scheme(url), _endpoint(url)
    if ep is None:
        return []
    if is_input and s in ("udp", "rtp"):
        return [ep[2], ep[2] + 1] if s == "rtp" else [ep[2]]
    if s == "srt" and any(k == "mode" and v == "listener" for k, v in _query_pairs(url)):
        return [ep[2]]
    return []


def _check_binds(inputs, outputs):
    """Du prisirišantys adresai tame pačiame vietiniame UDP porte — konfliktas (bind nepavyktų ar duomenys susimaišytų), nebent
    abu — skirtingos multicast grupės (pvz. config/examples/logo-udp-4.json: 4 grupės porte 5000)."""
    taken = []  # (portas, host, kelias, url)
    for prefix, items, is_input in (("inputs", inputs, True), ("outputs", outputs, False)):
        for k, it in enumerate(items):
            host = _endpoint(it.url)[1] if _endpoint(it.url) else ""
            for port in _local_ports(it.url, is_input):
                for p2, h2, path2, url2 in taken:
                    if p2 == port and not (_is_multicast(host) and _is_multicast(h2) and host != h2):
                        raise ConfigError(f"{prefix}[{k}].url", f"vietinis UDP portas {port} jau naudojamas: {path2} ({redact_url(url2)})")
                taken.append((port, host, f"{prefix}[{k}].url", it.url))


def _unique_endpoints(items, prefix, is_input):
    """Neprisirišantys adresai (siunčiantys išėjimai, srt caller, http): tikslus paskirties dublikatas — klaida."""
    seen = {}
    for k, it in enumerate(items):
        if _local_ports(it.url, is_input):
            continue  # prisirišantys — _check_binds
        ep = _endpoint(it.url)
        key = ep if ep is not None else it.url
        if key in seen:
            j = seen[key]
            raise ConfigError(f"{prefix}[{k}].url", f"tas pats adresas kaip {prefix}[{j}].url ({redact_url(it.url)})")
        seen[key] = k


# --- Sekcijos ----------------------------------------------------------------------------------------------------------

def _inputs(v):
    items = []
    for k, x in enumerate(_list(v, "inputs", 1, MAX_STREAMS, "įėjimų")):
        p = f"inputs[{k}]"
        _obj(x, p, ("url", "srt_passphrase_from"), ("url",))
        url = _url(x["url"], p + ".url", IN_SCHEMES)
        items.append(Input(url, _secret_ref(x.get("srt_passphrase_from"), p + ".srt_passphrase_from", url)))
    _unique_endpoints(items, "inputs", True)
    _same_secret(items, "inputs")
    return tuple(items)


def _outputs(v):
    items = []
    for k, x in enumerate(_list(v, "outputs", 1, MAX_STREAMS, "išėjimų")):
        p = f"outputs[{k}]"
        _obj(x, p, ("url", "vbitrate", "abitrate", "srt_passphrase_from"), ("url",))
        url = _url(x["url"], p + ".url", OUT_SCHEMES)
        vb = _int(x.get("vbitrate", DEFAULT_VBITRATE), p + ".vbitrate", *VBITRATE_RANGE)
        ab = _int(x.get("abitrate", DEFAULT_ABITRATE), p + ".abitrate", *ABITRATE_RANGE)
        items.append(Output(url, vb, ab, _secret_ref(x.get("srt_passphrase_from"), p + ".srt_passphrase_from", url)))
    _unique_endpoints(items, "outputs", False)
    _same_secret(items, "outputs")
    for k, it in enumerate(items[1:], 1):
        for f in ("vbitrate", "abitrate"):
            if getattr(it, f) != getattr(items[0], f):
                raise ConfigError(f"outputs[{k}].{f}", f"turi sutapti su outputs[0].{f} (vienas enc procesas koduoja visus srautus)")
    return tuple(items)


def png_size(path):
    """PNG matmenys iš IHDR (be bibliotekų); klaida — ValueError."""
    with open(path, "rb") as f:
        head = f.read(24)
    if len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
        raise ValueError("ne PNG failas")
    return struct.unpack(">II", head[16:24])


def _check_xy(x, y, size, frame, path, interlaced):
    (w, h), (W, H) = size, frame
    if x % 2:
        raise ConfigError(path + ".x", f"{x} — turi būti lyginis (4:2:2 chroma)")
    if interlaced and y % 2:
        raise ConfigError(path + ".y", f"{y} — interlaced formatui turi būti lyginis")
    if x + w > W:
        raise ConfigError(path + ".x", f"{x} + plotis {w} > kadro plotis {W}")
    if y + h > H:
        raise ConfigError(path + ".y", f"{y} + aukštis {h} > kadro aukštis {H}")


def _logo(v, frame, interlaced):
    p = "process.logo"
    if isinstance(v, str):
        if v != "builtin":
            raise ConfigError(p, "turi būti \"builtin\" arba objektas {\"file\": …, \"x\": …, \"y\": …}")
        return Logo()
    _obj(v, p, ("file", "x", "y"), ("file",))
    f = _str(v["file"], p + ".file")
    if f == "builtin":
        size = BUILTIN_LOGO
    else:
        try:
            size = png_size(f)
        except OSError as e:
            raise ConfigError(p + ".file", f"nepavyko atidaryti {f}: {e.strerror or e}") from None
        except ValueError as e:
            raise ConfigError(p + ".file", f"{f}: {e}") from None
        if size[0] % 2 or size[0] == 0 or size[1] == 0:
            raise ConfigError(p + ".file", f"{f}: logotipo plotis {size[0]} turi būti lyginis ir > 0")
    x = _int(v["x"], p + ".x", 0, frame[0]) if "x" in v else None
    y = _int(v["y"], p + ".y", 0, frame[1]) if "y" in v else None
    # numatytoji vieta — kaip logo.cpp: x = (W − w − 48) & ~1, y = 64
    _check_xy((frame[0] - size[0] - 48) & ~1 if x is None else x, 64 if y is None else y, size, frame, p, False)
    return Logo(f, x, y)


def _geom(g, path, frame):
    if g == "logo":
        return LAYER_LOGO
    if g == "full":
        return frame
    m = re.fullmatch(r"(\d+)x(\d+)", g) if isinstance(g, str) else None
    if not m or int(m.group(1)) % 2 or not 0 < int(m.group(1)) <= frame[0] or not 0 < int(m.group(2)) <= frame[1]:
        raise ConfigError(path, f"turi būti logo, full arba PLOTISxAUKŠTIS (lyginis plotis, ≤ {frame[0]}x{frame[1]})")
    return int(m.group(1)), int(m.group(2))


def _composite(v, frame, interlaced):
    p = "process.composite"
    _obj(v, p, ("layers",), ("layers",))
    layers = []
    for k, x in enumerate(_list(v["layers"], p + ".layers", 1, MAX_LAYERS, "sluoksnių")):
        lp = f"{p}.layers[{k}]"
        _obj(x, lp, ("source", "geom", "x", "y", "opacity"), ("source",))
        src = _str(x["source"], lp + ".source")
        if src not in SOURCES:
            raise ConfigError(lp + ".source", f"„{src}“ — palaikoma: {', '.join(SOURCES)}")
        geom = x.get("geom", "logo" if k == 0 else "full")
        size = _geom(geom, lp + ".geom", frame)
        lx = _int(x.get("x", 0), lp + ".x", 0, frame[0])
        ly = _int(x.get("y", 0), lp + ".y", 0, frame[1])
        op = x.get("opacity", 1)
        if isinstance(op, bool) or not isinstance(op, (int, float)) or not 0 <= op <= 1:
            raise ConfigError(lp + ".opacity", "turi būti skaičius 0…1")
        _check_xy(lx, ly, size, frame, lp, interlaced)
        layers.append(Layer(src, geom, lx, ly, op))
    return Composite(tuple(layers))


def _process(v, frame, interlaced):
    _obj(v, "process", ("logo", "composite"))
    kinds = [k for k in ("logo", "composite") if k in v]
    if len(kinds) != 1:
        raise ConfigError("process", "turi būti lygiai vienas iš: logo, composite" + (f" (rasta: {', '.join(kinds)})" if kinds else ""))
    if kinds[0] == "logo":
        return Process("logo", logo=_logo(v["logo"], frame, interlaced))
    return Process("composite", composite=_composite(v["composite"], frame, interlaced))


def parse(obj, source="") -> Plan:
    """Jau perskaitytas JSON (su _pairs) → Plan arba ConfigError."""
    _obj(obj, "", ("version", "format", "storage", "inputs", "process", "outputs", "restart", "input_wait_s"),
         ("version", "inputs", "process", "outputs"))
    if isinstance(obj["version"], bool) or obj["version"] != 1:
        raise ConfigError("version", f"nepalaikoma versija {json.dumps(obj['version'])} (palaikoma: 1)")
    fmt = obj.get("format", "1080i50")
    if fmt not in FORMATS:
        raise ConfigError("format", f"{json.dumps(fmt)} nepalaikomas (palaikoma: {', '.join(FORMATS)})")
    storage = obj.get("storage", "cuda-linear")
    if storage not in STORAGES:
        raise ConfigError("storage", f"{json.dumps(storage)} — palaikoma: {', '.join(STORAGES)}")
    frame = RASTER[re.match(r"\d+", fmt).group(0)]
    interlaced = "i" in fmt
    inputs = _inputs(obj["inputs"])
    process = _process(obj["process"], frame, interlaced)
    outputs = _outputs(obj["outputs"])
    _check_binds(inputs, outputs)
    if len(outputs) != len(inputs):
        raise ConfigError("outputs", f"{len(outputs)}, inputs: {len(inputs)} — turi sutapti")
    r = _obj(obj.get("restart", {}), "restart", ("max", "window_s"))
    restart = Restart(_int(r.get("max", 5), "restart.max", 0, 1000), _num(r.get("window_s", 60), "restart.window_s", 0, 86400))
    wait = _num(obj.get("input_wait_s", 20), "input_wait_s", 0, 3600)
    return Plan(1, fmt, storage, inputs, process, outputs, restart, wait, str(source))


def load(path) -> Plan:
    """Konfigūracijos failas → Plan. Klaida — ConfigError (kelias JSON'e ir priežastis)."""
    p = pathlib.Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise ConfigError("", f"nepavyko perskaityti {p}: {getattr(e, 'strerror', None) or e}") from None
    try:
        obj = json.loads(text, object_pairs_hook=_pairs)
    except json.JSONDecodeError as e:
        raise ConfigError("", f"netinkamas JSON: {e.msg} — eilutė {e.lineno}, stulpelis {e.colno}") from None
    return parse(obj, p)


def check_secrets(plan: Plan) -> None:
    """Slaptažodžių nuorodos pasiekiamos (prieš paleidžiant vaikus): env kintamasis nustatytas, failas skaitomas.
    Pati reikšmė neskaitoma į atmintį ilgiau nei patikrai ir niekur nerašoma."""
    for prefix, items in (("inputs", plan.inputs), ("outputs", plan.outputs)):
        for k, it in enumerate(items):
            ref = it.srt_passphrase_from
            if not ref:
                continue
            path = f"{prefix}[{k}].srt_passphrase_from"
            kind, _, what = ref.partition(":")
            if kind == "env" and not os.environ.get(what):
                raise ConfigError(path, f"aplinkos kintamasis {what} nenustatytas arba tuščias")
            if kind == "file":
                try:
                    ok = bool(pathlib.Path(what).read_bytes().strip())
                except OSError as e:
                    raise ConfigError(path, f"nepavyko perskaityti {what}: {e.strerror or e}") from None
                if not ok:
                    raise ConfigError(path, f"failas {what} tuščias")
