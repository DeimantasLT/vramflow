// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// nvenc_caps — klausia NVENC apie H.264 galimybes šioje kortoje ir praktiškai patikrina vienalaikių sesijų ribą.
// Svarbiausia: NV_ENC_CAPS_SUPPORT_FIELD_ENCODING (tikras interlaced H.264, 1080i50) ir kiek sesijų leidžia driver'is.
// Spausdina JSON. Naudoja ffnvcodec dynlink loader'į (libcuda + libnvidia-encode įkeliami dinamiškai).
#include <ffnvcodec/dynlink_loader.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

CudaFunctions* cu = nullptr;
NvencFunctions* nv = nullptr;
NV_ENCODE_API_FUNCTION_LIST api{};

void* open_session(CUcontext ctx, NVENCSTATUS* st) {
    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS p{};
    p.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
    p.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
    p.device = ctx;
    p.apiVersion = NVENCAPI_VERSION;
    void* enc = nullptr;
    *st = api.nvEncOpenEncodeSessionEx(&p, &enc);
    return *st == NV_ENC_SUCCESS ? enc : nullptr;
}

int cap(void* enc, NV_ENC_CAPS c) {
    NV_ENC_CAPS_PARAM p{};
    p.version = NV_ENC_CAPS_PARAM_VER;
    p.capsToQuery = c;
    int v = -1;
    if (api.nvEncGetEncodeCaps(enc, NV_ENC_CODEC_H264_GUID, &p, &v) != NV_ENC_SUCCESS) return -1;
    return v;
}

// Inicializuoja H.264 1920x1080 25 fps 8 Mbit CBR koderį (kaip realiame scenarijuje).
NVENCSTATUS init_h264(void* enc) {
    NV_ENC_PRESET_CONFIG preset{};
    preset.version = NV_ENC_PRESET_CONFIG_VER;
    preset.presetCfg.version = NV_ENC_CONFIG_VER;
    NVENCSTATUS st = api.nvEncGetEncodePresetConfigEx(enc, NV_ENC_CODEC_H264_GUID, NV_ENC_PRESET_P4_GUID,
                                                      NV_ENC_TUNING_INFO_LOW_LATENCY, &preset);
    if (st != NV_ENC_SUCCESS) return st;
    NV_ENC_CONFIG cfg = preset.presetCfg;
    cfg.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR;
    cfg.rcParams.averageBitRate = 8'000'000;
    NV_ENC_INITIALIZE_PARAMS ip{};
    ip.version = NV_ENC_INITIALIZE_PARAMS_VER;
    ip.encodeGUID = NV_ENC_CODEC_H264_GUID;
    ip.presetGUID = NV_ENC_PRESET_P4_GUID;
    ip.tuningInfo = NV_ENC_TUNING_INFO_LOW_LATENCY;
    ip.encodeWidth = ip.darWidth = 1920;
    ip.encodeHeight = ip.darHeight = 1080;
    ip.frameRateNum = 25;
    ip.frameRateDen = 1;
    ip.enablePTD = 1;
    ip.encodeConfig = &cfg;
    return api.nvEncInitializeEncoder(enc, &ip);
}

}  // namespace

int main(int argc, char** argv) {
    const int max_try = argc > 1 ? std::atoi(argv[1]) : 32;
    if (cuda_load_functions(&cu, nullptr) < 0 || nvenc_load_functions(&nv, nullptr) < 0) {
        std::printf("{\"error\":\"nepavyko įkelti libcuda/libnvidia-encode\"}\n");
        return 2;
    }
    uint32_t maxver = 0;
    nv->NvEncodeAPIGetMaxSupportedVersion(&maxver);
    api.version = NV_ENCODE_API_FUNCTION_LIST_VER;
    if (nv->NvEncodeAPICreateInstance(&api) != NV_ENC_SUCCESS) {
        std::printf("{\"error\":\"NvEncodeAPICreateInstance\",\"driver_max_api\":\"%u.%u\"}\n", maxver >> 4, maxver & 15);
        return 2;
    }
    CUdevice dev{};
    CUcontext ctx{};
    char name[128]{};
    if (cu->cuInit(0) || cu->cuDeviceGet(&dev, 0) || cu->cuCtxCreate(&ctx, 0, dev)) {
        std::printf("{\"error\":\"CUDA kontekstas\"}\n");
        return 2;
    }
    cu->cuDeviceGetName(name, sizeof name, dev);

    NVENCSTATUS st{};
    void* enc = open_session(ctx, &st);
    if (!enc) {
        std::printf("{\"error\":\"nvEncOpenEncodeSessionEx\",\"status\":%d}\n", st);
        return 2;
    }
    std::printf("{\"gpu\":\"%s\",\"driver_max_api\":\"%u.%u\",\"header_api\":\"%u.%u\",\n", name, maxver >> 4, maxver & 15,
                NVENCAPI_MAJOR_VERSION, NVENCAPI_MINOR_VERSION);
    std::printf(" \"h264_caps\":{\"field_encoding\":%d,\"max_bframes\":%d,\"width_max\":%d,\"height_max\":%d,"
                "\"async_encode\":%d,\"lossless\":%d,\"num_encoder_engines\":%d},\n",
                cap(enc, NV_ENC_CAPS_SUPPORT_FIELD_ENCODING), cap(enc, NV_ENC_CAPS_NUM_MAX_BFRAMES),
                cap(enc, NV_ENC_CAPS_WIDTH_MAX), cap(enc, NV_ENC_CAPS_HEIGHT_MAX), cap(enc, NV_ENC_CAPS_ASYNC_ENCODE_SUPPORT),
                cap(enc, NV_ENC_CAPS_SUPPORT_LOSSLESS_ENCODE), cap(enc, NV_ENC_CAPS_NUM_ENCODER_ENGINES));
    api.nvEncDestroyEncoder(enc);

    // Vienalaikių sesijų riba: atidarom ir inicializuojam, kol driver'is atsisako
    std::vector<void*> sessions;
    NVENCSTATUS last = NV_ENC_SUCCESS;
    const char* failed_at = "none";
    for (int i = 0; i < max_try; ++i) {
        void* e = open_session(ctx, &last);
        if (!e) { failed_at = "open"; break; }
        last = init_h264(e);
        if (last != NV_ENC_SUCCESS) {
            api.nvEncDestroyEncoder(e);
            failed_at = "initialize";
            break;
        }
        sessions.push_back(e);
    }
    std::printf(" \"concurrent_sessions_ok\":%zu,\"tried\":%d,\"failed_at\":\"%s\",\"last_status\":%d}\n", sessions.size(),
                max_try, failed_at, last);
    for (void* e : sessions) api.nvEncDestroyEncoder(e);
    cu->cuCtxDestroy(ctx);
    return 0;
}
