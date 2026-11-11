// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Pastovus gijų baseinas (CPU variantui ir gfx::cpu): atskirtas nuo func_common, kad bibliotekos jį naudotų be CUDA/MXL.
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace mxlv::func {

// Pastovus gijų baseinas CPU variantui: run(n, fn) iškviečia fn(i) visiems i < n ir laukia pabaigos.
// Gijos sukuriamos vieną kartą (ne kiekvienam tick'ui), kad kūrimo kaina nepatektų į matavimą.
class ThreadPool {
public:
    explicit ThreadPool(int threads);
    ~ThreadPool();
    template <class F>
    void run(int n, F&& fn) {
        job_ = [&fn](int i) { fn(i); };
        dispatch(n);
    }
    int size() const { return static_cast<int>(workers_.size()) + 1; }

private:
    void dispatch(int n);
    void worker();
    std::vector<std::thread> workers_;
    std::function<void(int)> job_;
    std::mutex m_;
    std::condition_variable cv_, done_cv_;
    std::atomic<int> next_{0};
    int n_ = 0, active_ = 0;
    std::uint64_t gen_ = 0;
    bool quit_ = false;
};

}  // namespace mxlv::func
