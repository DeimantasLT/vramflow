// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// cuda_fd_probe — žingsnis 0 (docs/architektura/2026-11-10-vramflow-verify.md §2).
// Tikrina, ar CUDA VMM atmintis su POSIX FD eksportu veikia tarp dviejų procesų šioje mašinoje
// (tas pats mechanizmas, kurį naudoja MXL PR #653 cuda-linear backend'as). Be MXL, tik driver API.
//
// Procesai: tėvas = importuotojas (reader), vaikas = eksportuotojas (writer). fork() prieš bet kokį
// CUDA kvietimą. Kiekvienas žingsnis spausdina vieną JSON eilutę į stdout.
// Išėjimo kodai: 0 PASS; 10 atributas 0; 11 create/export klaida; 12 import/map klaida;
//                13 duomenų neatitikimas; 14 nuotėkis; 1 kita klaida.
#include <cuda.h>

#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <dirent.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kBigBytes = 55'296'000;  // 10 × 1080p v210 grain'ų (5 529 600 B)
constexpr std::size_t kLoopBytes = 4u << 20;
constexpr int kLoopIters = 200;
constexpr std::int64_t kLeakSlackBytes = 16ll << 20;

const char* g_role = "?";

const char* curesult_name(CUresult r) {
    const char* n = nullptr;
    cuGetErrorName(r, &n);
    return n ? n : "UNKNOWN";
}

void jl(const char* step, const char* call, CUresult r, bool ok, const std::string& extra = {}) {
    std::printf("{\"role\":\"%s\",\"step\":\"%s\",\"call\":\"%s\",\"curesult\":%d,\"curesult_name\":\"%s\",\"ok\":%s%s%s}\n",
                g_role, step, call, static_cast<int>(r), curesult_name(r), ok ? "true" : "false",
                extra.empty() ? "" : ",", extra.c_str());
    std::fflush(stdout);
}

[[noreturn]] void fail(int code) {
    std::fflush(stdout);
    std::_Exit(code);
}

#define CU_STEP(step, call, code)                     \
    do {                                              \
        CUresult _r = (call);                         \
        jl(step, #call, _r, _r == CUDA_SUCCESS);      \
        if (_r != CUDA_SUCCESS) fail(code);           \
    } while (0)

// --- Unix socket žinutės: {type, status, size} + neprivalomas FD per SCM_RIGHTS -------------------
enum MsgType : std::int32_t { kFd = 1, kAck = 2, kExiting = 3, kAbort = 4 };
struct Msg {
    std::int32_t type;
    std::int32_t status;
    std::uint64_t size;
};

bool send_msg(int sock, const Msg& m, int fd = -1) {
    iovec iov{const_cast<Msg*>(&m), sizeof m};
    msghdr h{};
    h.msg_iov = &iov;
    h.msg_iovlen = 1;
    alignas(cmsghdr) char ctl[CMSG_SPACE(sizeof(int))]{};
    if (fd >= 0) {
        h.msg_control = ctl;
        h.msg_controllen = sizeof ctl;
        cmsghdr* c = CMSG_FIRSTHDR(&h);
        c->cmsg_level = SOL_SOCKET;
        c->cmsg_type = SCM_RIGHTS;
        c->cmsg_len = CMSG_LEN(sizeof(int));
        std::memcpy(CMSG_DATA(c), &fd, sizeof fd);
    }
    return sendmsg(sock, &h, 0) == static_cast<ssize_t>(sizeof m);
}

bool recv_msg(int sock, Msg& m, int* fd_out = nullptr) {
    iovec iov{&m, sizeof m};
    msghdr h{};
    h.msg_iov = &iov;
    h.msg_iovlen = 1;
    alignas(cmsghdr) char ctl[CMSG_SPACE(sizeof(int))]{};
    h.msg_control = ctl;
    h.msg_controllen = sizeof ctl;
    if (recvmsg(sock, &h, MSG_WAITALL) != static_cast<ssize_t>(sizeof m)) return false;
    if (fd_out) {
        *fd_out = -1;
        for (cmsghdr* c = CMSG_FIRSTHDR(&h); c; c = CMSG_NXTHDR(&h, c))
            if (c->cmsg_level == SOL_SOCKET && c->cmsg_type == SCM_RIGHTS) std::memcpy(fd_out, CMSG_DATA(c), sizeof(int));
    }
    return true;
}

int count_fds() {
    int n = 0;
    if (DIR* d = opendir("/proc/self/fd")) {
        while (dirent* e = readdir(d))
            if (e->d_name[0] != '.') ++n;
        closedir(d);
    }
    return n - 1;  // opendir pats atidaro vieną
}

// --- CUDA pagalbinės --------------------------------------------------------------------------------
CUdevice g_dev{};
CUcontext g_ctx{};
std::size_t g_gran = 0;

CUmemAllocationProp alloc_prop() {
    CUmemAllocationProp p{};
    p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    p.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    p.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    p.location.id = 0;
    return p;
}

std::size_t roundup(std::size_t v, std::size_t g) { return (v + g - 1) / g * g; }

void init_device(bool verbose) {
    CU_STEP("P0.1", cuInit(0), 1);
    int count = 0;
    CU_STEP("P0.1", cuDeviceGetCount(&count), 1);
    if (count < 1) fail(1);
    CU_STEP("P0.1", cuDeviceGet(&g_dev, 0), 1);
    if (verbose) {
        char name[256]{};
        cuDeviceGetName(name, sizeof name, g_dev);
        CUuuid u{};
        cuDeviceGetUuid(&u, g_dev);
        std::size_t total = 0;
        cuDeviceTotalMem(&total, g_dev);
        int drv = 0;
        cuDriverGetVersion(&drv);
        char uuid[40]{};
        for (int i = 0; i < 16; ++i) std::snprintf(uuid + 2 * i, 3, "%02x", static_cast<unsigned char>(u.bytes[i]));
        jl("P0.1", "device_info", CUDA_SUCCESS, true,
           "\"name\":\"" + std::string(name) + "\",\"uuid\":\"" + uuid + "\",\"total_mem\":" + std::to_string(total) +
               ",\"driver_cuda_version\":" + std::to_string(drv) + ",\"devices\":" + std::to_string(count));
    }
    CU_STEP("P0.4", cuDevicePrimaryCtxRetain(&g_ctx, g_dev), 1);
    CU_STEP("P0.4", cuCtxSetCurrent(g_ctx), 1);
    CUmemAllocationProp p = alloc_prop();
    CUresult r = cuMemGetAllocationGranularity(&g_gran, &p, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    jl("P0.4", "cuMemGetAllocationGranularity", r, r == CUDA_SUCCESS && g_gran > 0, "\"granularity\":" + std::to_string(g_gran));
    if (r != CUDA_SUCCESS || g_gran == 0) fail(11);
}

struct Mapping {
    CUmemGenericAllocationHandle h{};
    CUdeviceptr ptr{};
    std::size_t size{};
};

// Sukuria (arba importuoja) alokaciją ir ją sumapina READWRITE. `step` — žurnalui.
Mapping map_handle(CUmemGenericAllocationHandle h, std::size_t size, const char* step, int code, bool quiet) {
    Mapping m{h, 0, size};
    CUresult r = cuMemAddressReserve(&m.ptr, size, g_gran, 0, 0);
    if (!quiet || r) jl(step, "cuMemAddressReserve", r, r == CUDA_SUCCESS);
    if (r) fail(code);
    r = cuMemMap(m.ptr, size, 0, h, 0);
    if (!quiet || r) jl(step, "cuMemMap", r, r == CUDA_SUCCESS);
    if (r) fail(code);
    CUmemAccessDesc a{};
    a.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    a.location.id = 0;
    a.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    r = cuMemSetAccess(m.ptr, size, &a, 1);
    if (!quiet || r) jl(step, "cuMemSetAccess", r, r == CUDA_SUCCESS);
    if (r) fail(code);
    return m;
}

void unmap_release(Mapping& m) {
    cuMemUnmap(m.ptr, m.size);
    cuMemAddressFree(m.ptr, m.size);
    cuMemRelease(m.h);
    m = {};
}

void fill_pattern(std::vector<std::uint8_t>& b, unsigned mul, unsigned add) {
    for (std::size_t i = 0; i < b.size(); ++i) b[i] = static_cast<std::uint8_t>(i * mul + add);
}

std::size_t count_mismatch(const Mapping& m, unsigned mul, unsigned add) {
    std::vector<std::uint8_t> got(m.size);
    if (cuMemcpyDtoH(got.data(), m.ptr, m.size) != CUDA_SUCCESS) return SIZE_MAX;
    std::size_t bad = 0;
    for (std::size_t i = 0; i < got.size(); ++i) bad += got[i] != static_cast<std::uint8_t>(i * mul + add);
    return bad;
}

// --- Eksportuotojas (vaikas) ---------------------------------------------------------------------
[[noreturn]] void exporter(int sock) {
    g_role = "exporter";
    init_device(true);

    int vmm = 0, posix = 0;
    CUresult r = cuDeviceGetAttribute(&vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, g_dev);
    jl("P0.2", "VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED", r, r == CUDA_SUCCESS && vmm == 1, "\"value\":" + std::to_string(vmm));
    r = cuDeviceGetAttribute(&posix, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED, g_dev);
    jl("P0.3", "HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED", r, r == CUDA_SUCCESS && posix == 1, "\"value\":" + std::to_string(posix));
    if (vmm != 1 || posix != 1) { send_msg(sock, {kAbort, 10, 0}); fail(10); }

    // P0.14 (tik informacija): legacy IPC kelias
    {
        CUdeviceptr p{};
        if (cuMemAlloc(&p, 1 << 20) == CUDA_SUCCESS) {
            CUipcMemHandle ih{};
            CUresult ir = cuIpcGetMemHandle(&ih, p);
            jl("P0.14", "cuIpcGetMemHandle(info)", ir, true);
            cuMemFree(p);
        }
    }

    const std::size_t big = roundup(kBigBytes, g_gran);
    CUmemAllocationProp p = alloc_prop();
    CUmemGenericAllocationHandle h{};
    r = cuMemCreate(&h, big, &p, 0);
    jl("P0.5", "cuMemCreate", r, r == CUDA_SUCCESS, "\"size\":" + std::to_string(big));
    if (r) { send_msg(sock, {kAbort, 11, 0}); fail(11); }
    Mapping m = map_handle(h, big, "P0.5", 11, false);

    std::vector<std::uint8_t> host(big);
    fill_pattern(host, 131, 7);
    CU_STEP("P0.6", cuMemcpyHtoD(m.ptr, host.data(), big), 1);
    CU_STEP("P0.6", cuCtxSynchronize(), 1);

    int fd = -1;
    r = cuMemExportToShareableHandle(&fd, h, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0);
    struct stat st{};
    bool fd_ok = r == CUDA_SUCCESS && fd >= 0 && fstat(fd, &st) == 0;
    jl("P0.7", "cuMemExportToShareableHandle", r, fd_ok, "\"fd\":" + std::to_string(fd));
    if (!fd_ok) { send_msg(sock, {kAbort, 11, 0}); fail(11); }

    bool sent = send_msg(sock, {kFd, 0, big}, fd);
    jl("P0.8", "sendmsg(SCM_RIGHTS)", CUDA_SUCCESS, sent);
    close(fd);
    if (!sent) fail(1);

    Msg ack{};
    if (!recv_msg(sock, ack) || ack.type != kAck) fail(1);
    if (ack.status != 0) fail(ack.status);  // importuotojas jau pranešė priežastį

    std::size_t bad = count_mismatch(m, 17, 3);  // importuotojas perrašė
    jl("P0.11", "verify_importer_write", CUDA_SUCCESS, bad == 0, "\"mismatches\":" + std::to_string(bad));
    if (bad) { send_msg(sock, {kAbort, 13, 0}); fail(13); }

    // P0.13: ciklas create→export→send→(import→verify→release)→release
    const int fds0 = count_fds();
    std::size_t free0 = 0, total = 0;
    cuMemGetInfo(&free0, &total);
    const std::size_t small = roundup(kLoopBytes, g_gran);
    for (int it = 0; it < kLoopIters; ++it) {
        CUmemGenericAllocationHandle hh{};
        r = cuMemCreate(&hh, small, &p, 0);
        if (r) { jl("P0.13", "cuMemCreate(loop)", r, false, "\"iter\":" + std::to_string(it)); send_msg(sock, {kAbort, 11, 0}); fail(11); }
        Mapping mm = map_handle(hh, small, "P0.13", 11, true);
        cuMemsetD8(mm.ptr, static_cast<unsigned char>(it), small);
        cuCtxSynchronize();
        int f = -1;
        r = cuMemExportToShareableHandle(&f, hh, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0);
        if (r) { jl("P0.13", "cuMemExportToShareableHandle(loop)", r, false, "\"iter\":" + std::to_string(it)); send_msg(sock, {kAbort, 11, 0}); fail(11); }
        send_msg(sock, {kFd, it, small}, f);
        close(f);
        Msg a{};
        if (!recv_msg(sock, a) || a.type != kAck) fail(1);
        unmap_release(mm);
        if (a.status != 0) fail(a.status);
    }
    const int fds1 = count_fds();
    std::size_t free1 = 0;
    cuMemGetInfo(&free1, &total);
    const bool leak = fds1 != fds0 || static_cast<std::int64_t>(free1) < static_cast<std::int64_t>(free0) - kLeakSlackBytes;
    jl("P0.13", "leak_check", CUDA_SUCCESS, !leak,
       "\"fds_before\":" + std::to_string(fds0) + ",\"fds_after\":" + std::to_string(fds1) +
           ",\"free_before\":" + std::to_string(free0) + ",\"free_after\":" + std::to_string(free1));

    // P0.12: eksportuotojas atlaisvina savo pusę ir išeina; importuotojas toliau skaito
    send_msg(sock, {kExiting, leak ? 14 : 0, 0});
    unmap_release(m);
    cuDevicePrimaryCtxRelease(g_dev);
    jl("P0.12", "exporter_released_and_exiting", CUDA_SUCCESS, true);
    fail(leak ? 14 : 0);
}

// --- Importuotojas (tėvas) -----------------------------------------------------------------------
int importer(int sock, pid_t child) {
    g_role = "importer";
    init_device(false);

    Msg m{};
    int fd = -1;
    if (!recv_msg(sock, m, &fd)) return 1;
    if (m.type == kAbort) return m.status;
    jl("P0.8", "recvmsg(SCM_RIGHTS)", CUDA_SUCCESS, fd >= 0, "\"fd\":" + std::to_string(fd));
    if (fd < 0) return 1;

    CUmemGenericAllocationHandle h{};
    CUresult r = cuMemImportFromShareableHandle(&h, reinterpret_cast<void*>(static_cast<std::uintptr_t>(fd)),
                                                CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
    jl("P0.9", "cuMemImportFromShareableHandle", r, r == CUDA_SUCCESS);
    close(fd);
    if (r) { send_msg(sock, {kAck, 12, 0}); return 12; }
    Mapping big = map_handle(h, m.size, "P0.9", 12, false);

    std::size_t bad = count_mismatch(big, 131, 7);
    jl("P0.10", "verify_exporter_write", CUDA_SUCCESS, bad == 0, "\"mismatches\":" + std::to_string(bad) + ",\"bytes\":" + std::to_string(m.size));
    if (bad) { send_msg(sock, {kAck, 13, 0}); return 13; }

    std::vector<std::uint8_t> host(m.size);
    fill_pattern(host, 17, 3);
    if (cuMemcpyHtoD(big.ptr, host.data(), m.size) != CUDA_SUCCESS || cuCtxSynchronize() != CUDA_SUCCESS) {
        send_msg(sock, {kAck, 1, 0});
        return 1;
    }
    send_msg(sock, {kAck, 0, 0});

    const int fds0 = count_fds();
    for (;;) {
        Msg lm{};
        int f = -1;
        if (!recv_msg(sock, lm, &f)) return 1;
        if (lm.type == kAbort) return lm.status;
        if (lm.type == kExiting) {
            const int fds1 = count_fds();
            jl("P0.13", "importer_fd_check", CUDA_SUCCESS, fds0 == fds1,
               "\"fds_before\":" + std::to_string(fds0) + ",\"fds_after\":" + std::to_string(fds1));
            int status = 0;
            waitpid(child, &status, 0);
            usleep(500'000);
            std::size_t after = count_mismatch(big, 17, 3);
            jl("P0.12", "read_after_exporter_exit", CUDA_SUCCESS, after == 0, "\"mismatches\":" + std::to_string(after));
            unmap_release(big);
            const int child_code = WIFEXITED(status) ? WEXITSTATUS(status) : 1;
            if (after) return 13;
            if (child_code) return child_code;
            if (fds0 != fds1) return 14;
            return 0;
        }
        CUmemGenericAllocationHandle hh{};
        r = cuMemImportFromShareableHandle(&hh, reinterpret_cast<void*>(static_cast<std::uintptr_t>(f)),
                                           CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
        close(f);
        if (r) { jl("P0.13", "cuMemImportFromShareableHandle(loop)", r, false, "\"iter\":" + std::to_string(lm.status)); send_msg(sock, {kAck, 12, 0}); return 12; }
        Mapping mm = map_handle(hh, lm.size, "P0.13", 12, true);
        std::vector<std::uint8_t> got(lm.size);
        cuMemcpyDtoH(got.data(), mm.ptr, lm.size);
        std::size_t lbad = 0;
        for (auto v : got) lbad += v != static_cast<std::uint8_t>(lm.status);
        unmap_release(mm);
        if (lbad) { jl("P0.13", "verify(loop)", CUDA_SUCCESS, false, "\"iter\":" + std::to_string(lm.status)); send_msg(sock, {kAck, 13, 0}); return 13; }
        send_msg(sock, {kAck, 0, 0});
    }
}

}  // namespace

int main() {
    int sv[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) return 1;
    pid_t pid = fork();  // prieš bet kokį CUDA kvietimą
    if (pid < 0) return 1;
    if (pid == 0) {
        close(sv[0]);
        exporter(sv[1]);
    }
    close(sv[1]);
    const int code = importer(sv[0], pid);
    std::printf("{\"role\":\"importer\",\"step\":\"RESULT\",\"exit_code\":%d,\"pass\":%s}\n", code, code == 0 ? "true" : "false");
    return code;
}
