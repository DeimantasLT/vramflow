# Security

VRAMFlow is a proof of concept. There are no releases yet; fixes go to the `main` branch only.

## Reporting a problem

Use GitHub's private vulnerability reporting for this repository (**Security → Report a vulnerability**). Please do
not open a public issue for something that could be exploited. Include the commit, the command line and, if you can, a
small input that reproduces it.

## Trust model in short

The full threat model is in [`docs/architektura/2026-11-17-gresmiu-modelis.md`](docs/architektura/2026-11-17-gresmiu-modelis.md)
(Lithuanian). In short:

- **Network inputs are not trusted.** `ingest` reads them through libavformat with explicit protocol and format
  whitelists, playlist limits and a watchdog. This is defence in depth, not a sandbox: point `ingest` only at servers you
  operate or trust.
- **The MXL domain folder is the security boundary.** Whoever may write to the domain may write any content. VRAMFlow
  readers never trust sizes written by another process: a grain header or a flow description that does not match the
  storage is an error, not an out-of-bounds read.
- **Other local users are not trusted.** The lease folder next to the domain (`<domain>.vramflow`) is opened without
  following links and is refused if it belongs to someone else or is more permissive than it should be. Only those who may
  write to the domain can open its files, because whoever can open a lock file can also hold the lock.
- **Operator input is trusted but checked.** A malformed capacity file is rejected with the name of the field; it never
  silently disables a limit.

## Running it safely

- Run as a dedicated non-root user. Nothing needs root; `--rt-priority` needs `CAP_SYS_NICE` only. The whole test suite
  runs as a non-root user with `scripts/linux/45-nonroot.sh`.
- Keep the domain folder private to that user or to one group (`0700` or `0770`). Never use a world-writable domain on a
  machine with other users.
- Put the domain into a parent folder that only that user or group can write to, for example `/dev/shm/vramflow/<domain>`.
  Directly in `/dev/shm` another local user can create `<domain>.vramflow` first; VRAMFlow then refuses to start (exit
  code `2`) rather than use a folder it does not trust.
- Keep the capacity file writable by the operator only.
- With systemd, a unit along these lines keeps a compromised process contained:
  `User=`, `NoNewPrivileges=yes`, `ProtectSystem=strict`, `ProtectHome=yes`, `PrivateTmp=yes`,
  `ReadWritePaths=/dev/shm` plus the output folder, and `DeviceAllow=` for the GPU devices.
- In a container: no `--privileged`, no extra capabilities, a read-only root file system. Share the **parent** folder that
  holds both `<domain>` and `<domain>.vramflow`, not the domain alone: the lease folder sits next to the domain, so a volume
  mounted at the domain path leaves every container with its own leases and its own admission ledger (or, on a read-only
  root, with none: exit code `2`). VRAM flows between containers have not been tested yet.
- The binaries are built with stack protector, `_FORTIFY_SOURCE=3`, full RELRO, PIE and a non-executable stack
  (`ctest build_hardening` checks every binary). Keep `VRAMFLOW_HARDENING=ON`.

## What is checked automatically

`ctest` covers: build hardening, lease and ledger file safety, hostile writers, the URL and playlist policy, packet queue
limits, the libav watchdog, capacity file validation, and the regression corpus of the parser fuzzers
(`tests/fuzz/corpus/`). `scripts/linux/56-fuzz.sh` runs the fuzzers themselves.
