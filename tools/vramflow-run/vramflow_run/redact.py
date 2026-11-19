# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""URL be paslapčių žurnalams ir pranešimams — ta pati elgsena kaip C++ `mxlv::ingest::redact_url`
(src/ingest/url_policy.cpp; tie patys atvejai — test_config.py Redact ir tests/unit/test_url_policy.cpp).

userinfo → „***@“, kiekvienos užklausos poros reikšmė → „***“ (raktas paliekamas), fragmentas → „#***“; vietinis kelias —
nepakeistas; schema be „://“ iš karto po jos — „schema:***“.
"""
import string

_SCHEME_CHARS = set(string.ascii_letters + string.digits + "+-.")


def url_scheme(url: str) -> str:
    """Schema mažosiomis raidėmis arba "" (vietinis kelias), kaip C++ url_scheme (be „subfile,“ atvejo — jis čia nenaudojamas)."""
    n = 0
    while n < len(url) and url[n] in _SCHEME_CHARS:
        n += 1
    if n == 0 or n >= len(url) or url[n] != ":":
        return ""
    return url[:n].lower()


def _first_of(s: str, chars: str, start: int) -> int:
    idx = [i for i in (s.find(c, start) for c in chars) if i >= 0]
    return min(idx) if idx else len(s)


def redact_url(url: str) -> str:
    s = url_scheme(url)
    if not s:
        return url
    if url[len(s):len(s) + 3] != "://":
        return s + ":***"  # kaip C++: schema mažosiomis
    auth0 = len(s) + 3
    out = url[:auth0]
    auth1 = _first_of(url, "/?#", auth0)
    auth = url[auth0:auth1]
    at = auth.rfind("@")
    out += auth if at < 0 else "***@" + auth[at + 1:]
    q = _first_of(url, "?#", auth1)
    out += url[auth1:q]
    if q < len(url) and url[q] == "?":
        f = url.find("#", q)
        f = len(url) if f < 0 else f
        parts = []
        for kv in url[q + 1:f].split("&"):
            eq = kv.find("=")
            parts.append(("***" if kv else "") if eq < 0 else kv[:eq + 1] + "***")
        out += "?" + "&".join(parts)
        if f < len(url):
            out += "#***"
    elif q < len(url):
        out += "#***"
    return out
