// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
#include "func/thread_pool.hpp"

namespace mxlv::func {

ThreadPool::ThreadPool(int threads) {
    for (int t = 1; t < threads; ++t) workers_.emplace_back([this] { worker(); });  // kviečianti gija — taip pat darbininkas
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard g(m_);
        quit_ = true;
    }
    cv_.notify_all();
    for (auto& w : workers_) w.join();
}

void ThreadPool::dispatch(int n) {
    {
        std::lock_guard g(m_);
        n_ = n;
        next_ = 0;
        active_ = static_cast<int>(workers_.size());
        ++gen_;
    }
    cv_.notify_all();
    for (int i; (i = next_.fetch_add(1)) < n;) job_(i);
    std::unique_lock lk(m_);
    done_cv_.wait(lk, [this] { return active_ == 0; });
}

void ThreadPool::worker() {
    std::uint64_t seen = 0;
    for (;;) {
        int n = 0;
        {
            std::unique_lock lk(m_);
            cv_.wait(lk, [&] { return quit_ || gen_ != seen; });
            if (quit_) return;
            seen = gen_;
            n = n_;
        }
        for (int i; (i = next_.fetch_add(1)) < n;) job_(i);
        {
            std::lock_guard g(m_);
            if (--active_ == 0) done_cv_.notify_one();
        }
    }
}

}  // namespace mxlv::func
