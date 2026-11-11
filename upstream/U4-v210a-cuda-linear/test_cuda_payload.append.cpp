// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// Appended to lib/tests/cuda/test_cuda_payload.cpp (PR #653 @ 6dddc91d92f5). Uses Domain, deviceOptions, readFile and the
// includes of that file; <algorithm> and <random> are added to its include list. Uses the existing data/v210a_flow.json (1080p).

namespace
{
    /** Id of the flow in data/v210a_flow.json. */
    constexpr auto V210A_FLOW_ID = "5fbec3b1-1b0f-417d-9059-8b94a47197ed";

    /** Lines of the 1080p flow in data/v210a_flow.json. */
    constexpr auto V210A_LINES = std::size_t{1080};

    /** Bytes per line of the fill plane (v210, 1920 pixels) and of the key plane (3 pixels in 32 bits). */
    constexpr auto V210A_FILL_PITCH = std::size_t{5120};
    constexpr auto V210A_KEY_PITCH = std::size_t{2560};

    /**
     * A writer and a reader of the v210a test flow whose payload is in CUDA device memory, with the slots of both mapped.
     */
    class V210aFlow
    {
    public:
        /** \param[in] domain The MXL domain. */
        explicit V210aFlow(std::filesystem::path const& domain)
            : _instance{mxlCreateInstance(domain.string().c_str(), "")}
        {
            // A failed REQUIRE throws, and the destructor does not run: release what exists so far.
            try
            {
                REQUIRE(_instance != nullptr);
                auto const flowDef = readFile(std::filesystem::path{MXL_TEST_DATA_DIR} / "v210a_flow.json");
                auto const options = deviceOptions(mxl::tests::cuda::deviceUuid(0));
                REQUIRE(mxlCreateFlowWriter(_instance, flowDef.c_str(), options.c_str(), &_writer, &_configInfo, nullptr) == MXL_STATUS_OK);
                REQUIRE(mxlFlowWriterGetStorageLayout(_writer, &_layout) == MXL_STATUS_OK);
                _writerSlots.resize(_layout.slotCount);
                REQUIRE(mxlFlowWriterMapSlots(_writer, _layout.slotCount, _writerSlots.data()) == MXL_STATUS_OK);

                REQUIRE(mxlCreateFlowReader(_instance, V210A_FLOW_ID, "", &_reader) == MXL_STATUS_OK);
                _readerSlots.resize(_layout.slotCount);
                REQUIRE(mxlFlowReaderMapSlots(_reader, _layout.slotCount, _readerSlots.data()) == MXL_STATUS_OK);
            }
            catch (...)
            {
                release();
                throw;
            }
        }

        /** Release the reader and the writer before the instance, which deletes the flow. */
        ~V210aFlow()
        {
            release();
        }

        V210aFlow(V210aFlow const&) = delete;
        V210aFlow(V210aFlow&&) = delete;
        V210aFlow& operator=(V210aFlow const&) = delete;
        V210aFlow& operator=(V210aFlow&&) = delete;

        /** Release the reader and the writer, then destroy the instance. */
        void release() noexcept
        {
            if (_reader != nullptr)
            {
                mxlReleaseFlowReader(_instance, _reader);
                _reader = nullptr;
            }
            if (_writer != nullptr)
            {
                mxlReleaseFlowWriter(_instance, _writer);
                _writer = nullptr;
            }
            if (_instance != nullptr)
            {
                mxlDestroyInstance(_instance);
                _instance = nullptr;
            }
        }

        /** \return The writer. */
        [[nodiscard]]
        mxlFlowWriter writer() const noexcept
        {
            return _writer;
        }

        /** \return The reader. */
        [[nodiscard]]
        mxlFlowReader reader() const noexcept
        {
            return _reader;
        }

        /** \return The storage layout of the flow. */
        [[nodiscard]]
        mxlGrainStorageLayout const& layout() const noexcept
        {
            return _layout;
        }

        /** \return The configuration of the flow. */
        [[nodiscard]]
        mxlFlowConfigInfo const& configInfo() const noexcept
        {
            return _configInfo;
        }

        /**
         * \param[in] slot A slot number.
         * \param[in] plane A plane number.
         * \return The device address of the plane in the writer's mapping.
         */
        [[nodiscard]]
        std::uint8_t* writerPlane(std::uint32_t slot, std::size_t plane) const noexcept
        {
            return static_cast<std::uint8_t*>(mxlCudaLinearDevicePointer(_writerSlots[slot])) + _layout.planes[plane].offset;
        }

        /**
         * \param[in] slot A slot number.
         * \param[in] plane A plane number.
         * \return The device address of the plane in the reader's mapping.
         */
        [[nodiscard]]
        std::uint8_t const* readerPlane(std::uint32_t slot, std::size_t plane) const noexcept
        {
            return static_cast<std::uint8_t const*>(mxlCudaLinearDevicePointer(_readerSlots[slot])) + _layout.planes[plane].offset;
        }

    private:
        mxlInstance _instance{nullptr};            ///< The instance of the writer and the reader.
        mxlFlowWriter _writer{nullptr};            ///< The writer.
        mxlFlowReader _reader{nullptr};            ///< The reader.
        mxlFlowConfigInfo _configInfo{};           ///< Configuration of the created flow.
        mxlGrainStorageLayout _layout{};           ///< Storage layout of the flow.
        std::vector<mxlGrainStorage> _writerSlots; ///< The writer's slots.
        std::vector<mxlGrainStorage> _readerSlots; ///< The reader's slots.
    };

    /**
     * \param[in] size A number of bytes.
     * \param[in] seed The seed of the pseudo random sequence.
     * \return Pseudo random bytes.
     */
    std::vector<std::uint8_t> randomBytes(std::size_t size, std::uint32_t seed)
    {
        auto generator = std::mt19937{seed};
        auto bytes = std::vector<std::uint8_t>(size);
        std::generate(bytes.begin(), bytes.end(), [&generator] { return static_cast<std::uint8_t>(generator()); });
        return bytes;
    }

    /**
     * Copy bytes from device memory.
     * \param[in] pointer A device address.
     * \param[in] size The number of bytes.
     * \return The bytes.
     */
    std::vector<std::uint8_t> copyFromDevice(void const* pointer, std::size_t size)
    {
        auto bytes = std::vector<std::uint8_t>(size);
        REQUIRE(::cudaMemcpy(bytes.data(), pointer, size, cudaMemcpyDeviceToHost) == cudaSuccess);
        return bytes;
    }
}

///
/// A v210a flow in cuda-linear storage has two planes: the fill (v210), then the key (10 bits per pixel, 3 pixels in 32 bits).
///
TEST_CASE("CUDA payload : v210a has a fill plane and a key plane", "[cuda payload]")
{
    if (mxl::tests::cuda::deviceCount() == 0)
    {
        SKIP("No CUDA device.");
    }
    auto const domain = Domain{};
    auto const flow = V210aFlow{domain.path()};

    auto const& layout = flow.layout();
    REQUIRE(std::string_view{layout.storageType} == MXL_CUDA_LINEAR_STORAGE_TYPE);
    REQUIRE(layout.planeCount == 2U);
    REQUIRE(layout.planes[0].size == V210A_FILL_PITCH * V210A_LINES);
    REQUIRE(layout.planes[1].size == V210A_KEY_PITCH * V210A_LINES);
    REQUIRE(layout.planes[0].pitch == V210A_FILL_PITCH);
    REQUIRE(layout.planes[1].pitch == V210A_KEY_PITCH);
    REQUIRE(layout.planes[0].offset == 0U);
    REQUIRE(layout.planes[1].offset == layout.planes[0].size);

    REQUIRE(flow.configInfo().discrete.sliceSizes[0] == V210A_FILL_PITCH);
    REQUIRE(flow.configInfo().discrete.sliceSizes[1] == V210A_KEY_PITCH);

    auto const rate = flow.configInfo().common.grainRate;
    auto grainInfo = mxlGrainInfo{};
    auto slot = std::uint32_t{0};
    REQUIRE(mxlFlowWriterOpenGrainSlot(flow.writer(), mxlGetCurrentIndex(&rate), &grainInfo, &slot) == MXL_STATUS_OK);
    REQUIRE(grainInfo.grainSize == layout.planes[0].size + layout.planes[1].size);
    REQUIRE(grainInfo.totalSlices == V210A_LINES);
    REQUIRE(mxlFlowWriterCancelGrain(flow.writer()) == MXL_STATUS_OK);
}

///
/// Both planes of a v210a grain written to device memory arrive unchanged at a reader.
///
TEST_CASE("CUDA payload : v210a fill and key planes reach the reader", "[cuda payload]")
{
    if (mxl::tests::cuda::deviceCount() == 0)
    {
        SKIP("No CUDA device.");
    }
    auto const domain = Domain{};
    auto const flow = V210aFlow{domain.path()};
    auto const& layout = flow.layout();
    REQUIRE(layout.planeCount == 2U);

    auto const fill = randomBytes(layout.planes[0].size, 1U);
    auto const key = randomBytes(layout.planes[1].size, 2U);

    auto const rate = flow.configInfo().common.grainRate;
    auto const index = mxlGetCurrentIndex(&rate);
    auto grainInfo = mxlGrainInfo{};
    auto writerSlot = std::uint32_t{0};
    REQUIRE(mxlFlowWriterOpenGrainSlot(flow.writer(), index, &grainInfo, &writerSlot) == MXL_STATUS_OK);
    REQUIRE(::cudaMemcpy(flow.writerPlane(writerSlot, 0), fill.data(), fill.size(), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(::cudaMemcpy(flow.writerPlane(writerSlot, 1), key.data(), key.size(), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(::cudaDeviceSynchronize() == cudaSuccess);
    grainInfo.validSlices = grainInfo.totalSlices;
    REQUIRE(mxlFlowWriterCommitGrain(flow.writer(), &grainInfo) == MXL_STATUS_OK);

    auto readInfo = mxlGrainInfo{};
    auto readerSlot = std::uint32_t{0};
    REQUIRE(mxlFlowReaderGetGrainSlot(flow.reader(), index, MXL_GRAIN_VALID_SLICES_ALL, 0U, &readInfo, &readerSlot) == MXL_STATUS_OK);
    REQUIRE(readerSlot == writerSlot);
    REQUIRE(readInfo.validSlices == readInfo.totalSlices);
    REQUIRE(copyFromDevice(flow.readerPlane(readerSlot, 0), fill.size()) == fill);
    REQUIRE(copyFromDevice(flow.readerPlane(readerSlot, 1), key.size()) == key);
}

///
/// Partial grains of a v210a flow: a slice is one line of each plane, so after a commit of N slices a reader that asks for
/// at most N slices sees N lines of the fill and N lines of the key, and a reader that asks for N + 1 slices is told that
/// it is too early.
///
TEST_CASE("CUDA payload : v210a partial grains cover both planes", "[cuda payload]")
{
    if (mxl::tests::cuda::deviceCount() == 0)
    {
        SKIP("No CUDA device.");
    }
    auto const domain = Domain{};
    auto const flow = V210aFlow{domain.path()};
    auto const& layout = flow.layout();
    REQUIRE(layout.planeCount == 2U);

    auto const fill = randomBytes(layout.planes[0].size, 3U);
    auto const key = randomBytes(layout.planes[1].size, 4U);

    auto const rate = flow.configInfo().common.grainRate;
    auto const index = mxlGetCurrentIndex(&rate);
    auto grainInfo = mxlGrainInfo{};
    auto writerSlot = std::uint32_t{0};
    REQUIRE(mxlFlowWriterOpenGrainSlot(flow.writer(), index, &grainInfo, &writerSlot) == MXL_STATUS_OK);
    REQUIRE(grainInfo.totalSlices == V210A_LINES);

    auto const batch = static_cast<std::uint16_t>(grainInfo.totalSlices / 4U);
    for (auto committed = std::uint16_t{0}; committed < grainInfo.totalSlices;)
    {
        auto const next = static_cast<std::uint16_t>(committed + batch);
        // The lines [committed, next) of both planes.
        REQUIRE(::cudaMemcpy(flow.writerPlane(writerSlot, 0) + committed * V210A_FILL_PITCH,
                    fill.data() + committed * V210A_FILL_PITCH,
                    (next - committed) * V210A_FILL_PITCH,
                    cudaMemcpyHostToDevice) == cudaSuccess);
        REQUIRE(::cudaMemcpy(flow.writerPlane(writerSlot, 1) + committed * V210A_KEY_PITCH,
                    key.data() + committed * V210A_KEY_PITCH,
                    (next - committed) * V210A_KEY_PITCH,
                    cudaMemcpyHostToDevice) == cudaSuccess);
        REQUIRE(::cudaDeviceSynchronize() == cudaSuccess);
        grainInfo.validSlices = next;
        REQUIRE(mxlFlowWriterCommitGrain(flow.writer(), &grainInfo) == MXL_STATUS_OK);
        committed = next;

        auto readInfo = mxlGrainInfo{};
        auto readerSlot = std::uint32_t{0};
        if (committed < grainInfo.totalSlices)
        {
            REQUIRE(mxlFlowReaderGetGrainSlot(flow.reader(), index, static_cast<std::uint16_t>(committed + 1U), 0U, &readInfo, &readerSlot) ==
                    MXL_ERR_OUT_OF_RANGE_TOO_EARLY);
        }
        REQUIRE(mxlFlowReaderGetGrainSlot(flow.reader(), index, committed, 0U, &readInfo, &readerSlot) == MXL_STATUS_OK);
        REQUIRE(readerSlot == writerSlot);
        REQUIRE(readInfo.validSlices == committed);

        auto const fillLines = copyFromDevice(flow.readerPlane(readerSlot, 0), committed * V210A_FILL_PITCH);
        auto const keyLines = copyFromDevice(flow.readerPlane(readerSlot, 1), committed * V210A_KEY_PITCH);
        REQUIRE(std::equal(fillLines.begin(), fillLines.end(), fill.begin()));
        REQUIRE(std::equal(keyLines.begin(), keyLines.end(), key.begin()));
    }

    auto readInfo = mxlGrainInfo{};
    auto readerSlot = std::uint32_t{0};
    REQUIRE(mxlFlowReaderGetGrainSlot(flow.reader(), index, MXL_GRAIN_VALID_SLICES_ALL, 0U, &readInfo, &readerSlot) == MXL_STATUS_OK);
    REQUIRE(readInfo.validSlices == readInfo.totalSlices);
}
