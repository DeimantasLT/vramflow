<!-- SPDX-FileCopyrightText: 2026 Deimantas Kiršinas -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# OGraf → `video/v210a` example

An [EBU OGraf](https://github.com/ebu/ograf) graphic (HTML web component) is rendered to RGBA frames with real alpha,
fed through the public `libvramflow-gfx` library (`vfgfx_submit`, pacing mode) into a standard `video/v210a` MXL flow,
and put over a `gen` video with `vramflow-func composite`. This example is separate from the library and the compositor;
it uses only their public interfaces (`include/vramflow/gfx.h`, MXL C API).

```
renderer (Edge/CDP, WPE, CEF, …) --RGBA/BGRA frames--> ograf_feed --vfgfx_submit--> v210a flow --> composite --> v210 / v210a
```

## Files

| File | What |
|---|---|
| `ograf_feed.cpp` | Adapter. `--source stdin` reads raw frames (1920×1080×4 B, straight alpha, `--format rgba\|bgra`); `--source synthetic` is a built-in animated source with soft alpha (no renderer needed). Starts the library's pacing thread and calls `vfgfx_submit` for every new frame; the library commits the latest frame every field. |
| `ograf_probe.cpp` | Check: reads the `v210a` output of `composite`, counts key values, saves one field as PNG (key and Y). |
| `render_edge.mjs` | Renderer: headless Microsoft Edge driven over the Chrome DevTools Protocol (Node ≥ 22, no packages), transparent page background, frames → stdout as RGBA. |
| `graphic/` | A small hand-written OGraf v1 graphic (`lower-third.ograf.json` + web component) and a minimal host page. Synthetic, **not** one of the EBU sample graphics. |
| `run-demo.sh` | Builds the two programs and runs the whole demo in WSL. |

## 11.1 Research: what can render HTML with alpha, headless, today

Checked on 2026-11-11 on this workstation (nothing was downloaded or installed):

| Candidate | Result |
|---|---|
| WSL Ubuntu 24.04, GStreamer 1.24.2: `gst-inspect-1.0 wpesrc` / `cefsrc` | **not installed** ("No such element or plugin") |
| WSL: chromium, chromium-browser, google-chrome, firefox, headless_shell, wkhtmltoimage, cutycapt | not installed |
| WSL Python: PIL, cairo, playwright, cv2 | not installed (numpy 1.26.4 only) |
| WSL Node | not installed (Windows Node is reachable through interop) |
| Windows: Chrome | not installed |
| **Windows: Microsoft Edge 157.0.4341.42** (`C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe`) + **Node v24.22.0** (built-in `WebSocket`) | **usable** |

Edge is Chromium. Headless with `Emulation.setDefaultBackgroundColorOverride {r:0,g:0,b:0,a:0}` and
`Page.captureScreenshot` (PNG, straight alpha) gives a really transparent page. Measured with the OGraf graphic in
`graphic/`, 1920×1080 frame: 1 723 654 pixels alpha 0, 6 350 pixels alpha 255, 343 596 pixels with intermediate alpha
(the drop shadow and anti-aliased edges), identical in the first and last frame of a 4 s run.

Limits of this renderer: screenshot-based, so ~4–5 fps at 1080p here. This is enough for a graphic that animates in and
then stays (pacing mode holds the last frame and commits it every field), but not for smooth continuous animation. The
Windows → WSL hop is a plain pipe (`node.exe … | ograf_feed`).

### OGraf, as of this research (read-only web lookup)

- Spec: OGraf v1 (Graphics specification published 2025-09-17; Server API specification listed 2026-08-13), repo
  `ebu/ograf`, sample graphics in `v1/examples` (`bar-chart`, `headline`, `l3rd-name`, `minimal`, `ograf-logo`,
  `renderer-test`, `responsive-lower-third`, `scoreboard`, `weather`).
- A graphic = a manifest `*.ograf.json` (`$schema`, `id`, `name`, `main`, `supportsRealTime`, `supportsNonRealTime`, …)
  plus an ES module whose default export is a web component with `load({data, renderType, renderCharacteristics})`,
  `dispose`, `playAction`, `stopAction`, `updateAction`, `customAction` (and `goToTime`, `setActionsSchedule` for
  non-real-time). The specification text I read does not define a background/transparency requirement; transparency is
  the renderer's job. `graphic/host.html` is a minimal host: manifest → import → `load` → `playAction`.
- Renderers named by the project: OGraf Devtool (SuperFlyTV/ograf-devtool) and OGraf Simple Rendering System
  (SuperFlyTV/ograf-server). Neither is documented as headless-with-alpha; not evaluated.

## 11.2 Run

```bash
# WSL, repo built with scripts/wsl/30-build-verify.sh -DMXLV_BUILD_FUNC=ON
SOURCE=edge      bash examples/ograf/run-demo.sh /tmp/vf-ograf   # live OGraf render through Edge (needs Windows Edge + Node)
SOURCE=synthetic bash examples/ograf/run-demo.sh /tmp/vf-ograf   # built-in source, nothing else needed
```

The script starts `gen` (2 flows), `ograf_feed`, then two compositors on the same graphics flow:

- **A**: `composite --base gen --out-format v210` — the program picture (graphic over video). The base is opaque, so the
  key of a `v210a` output over a `gen` base would be 940 everywhere by definition; that is why A is `v210`.
- **B**: `composite --base none --out-format v210a` — the graphic alone, alpha preserved; `ograf_probe` checks this one.

Result of the runs on 2026-11-11 (RTX 4070 Laptop, WSL2, `--impl cpu --storage host`, 20–24 s, power profile "Whisper"):

| Source | Frames submitted | Probe (300 fields of the `v210a` output) |
|---|---|---|
| `edge` (OGraf lower third) | 88 (4.0 fps) | key = 64: 259 409 930; key = 940: 955 670; **64 < key < 940: 51 711 200** (of 312 076 800) |
| `synthetic` | 528 | key = 64: 221 173 443; key = 940: 82 153; **64 < key < 940: 90 821 204** (of 312 076 800) |

The output is committed (both compositors committed ~950 fields in their window) and the key has intermediate values, so
the alpha passes through the library and the compositor. One field is saved as `ograf-key.png` and `ograf-fill-y.png`
(grayscale, 1920×540, in the output directory only; not committed). Both compositors report exit code 4
(capacity event: 38–45 missing fields of ~1000, ~4 %); `verify` does not apply (the content is not deterministic) and
this example makes no capacity claim. Not investigated further.

## Other renderers (not run)

`ograf_feed --source stdin --format bgra` accepts what a GStreamer pipeline writes, for example
`gst-launch-1.0 wpesrc location=<url> ! video/x-raw,format=BGRA,width=1920,height=1080 ! fdsink fd=1 | ograf_feed …`
(whether `wpesrc` output is premultiplied was not checked; if it is, add `--alpha premultiplied`). This path was **not run**
here because `wpesrc`/`cefsrc` are not installed.

To run the official EBU sample graphics or a smooth GStreamer path the author would need to approve (each needs a
download, which this work did not do):

- `git clone https://github.com/ebu/ograf` (sample graphics; serve `v1/examples/<name>` over HTTP and point `host.html`
  at its manifest), and/or
- `sudo apt install gstreamer1.0-wpe` in WSL (package name to be checked; provides `wpesrc`).
