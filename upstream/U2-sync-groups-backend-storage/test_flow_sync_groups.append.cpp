
namespace
{
    /// Writer options that move the grain payload of a discrete flow to the test payload backend.
    constexpr auto TEST_BACKEND_OPTIONS = R"({"payload": {"backend": "test-file"}})";

    /// Commit the grain with the given index through the slot descriptor API after `delay`.
    ///
    /// The pointer based API (mxlFlowWriterOpenGrain) returns MXL_ERR_UNSUPPORTED_OPERATION for payload
    /// backends, so flows with backend payload storage are written through mxlFlowWriterOpenGrainSlot. The
    /// payload itself is not touched: synchronization only depends on the committed grain header.
    mxlStatus commitGrainSlotAfter(mxlFlowWriter writer, std::uint64_t index, std::chrono::milliseconds delay)
    {
        std::this_thread::sleep_for(delay);

        auto info = mxlGrainInfo{};
        auto slot = std::uint32_t{0};
        if (auto const status = mxlFlowWriterOpenGrainSlot(writer, index, &info, &slot); status != MXL_STATUS_OK)
        {
            return status;
        }
        info.validSlices = info.totalSlices;
        return mxlFlowWriterCommitGrain(writer, &info);
    }
}

///
/// A synchronization group must wait for members whose grain payload lives in a payload backend.
///
/// The group only reads grain headers (FlowSynchronizationGroup::waitForDataAt calls the reader with no
/// payload output), so a member with backend payload storage must behave exactly like a member with host
/// storage. This test makes the backend member the slow one: the wait must not complete before it commits,
/// and a wait for a grain that nobody writes must time out.
///
TEST_CASE_PERSISTENT_FIXTURE(mxl::tests::mxlDomainFixture, "Synchronization group : Member with backend payload storage",
    "[mxl sync groups][grain storage]")
{
    auto instance = mxlCreateInstance(domain.string().c_str(), "{}");
    REQUIRE(instance != nullptr);

    auto videoWriter = mxlFlowWriter{nullptr};
    auto dataWriter = mxlFlowWriter{nullptr};
    auto configInfo = mxlFlowConfigInfo{};
    auto created = false;
    REQUIRE(mxlCreateFlowWriter(instance, mxl::tests::readFile("data/v210_flow.json").c_str(), TEST_BACKEND_OPTIONS, &videoWriter,
                &configInfo, &created) == MXL_STATUS_OK);
    REQUIRE(created);
    REQUIRE(
        mxlCreateFlowWriter(instance, mxl::tests::readFile("data/data_flow.json").c_str(), "", &dataWriter, &configInfo, &created) == MXL_STATUS_OK);

    auto videoReader = mxlFlowReader{nullptr};
    auto dataReader = mxlFlowReader{nullptr};
    REQUIRE(mxlCreateFlowReader(instance, VIDEO_FLOW_ID, "", &videoReader) == MXL_STATUS_OK);
    REQUIRE(mxlCreateFlowReader(instance, DATA_FLOW_ID, "", &dataReader) == MXL_STATUS_OK);

    // The video flow really uses the backend: backend flows use flow data version 2 (the test backend's slots
    // are host pointers into its own file, so the storage type alone does not tell them apart).
    auto videoInfo = mxlFlowInfo{};
    REQUIRE(mxlFlowReaderGetInfo(videoReader, &videoInfo) == MXL_STATUS_OK);
    REQUIRE(videoInfo.version == 2U);

    auto group = mxlFlowSynchronizationGroup{nullptr};
    REQUIRE(mxlCreateFlowSynchronizationGroup(instance, &group) == MXL_STATUS_OK);
    REQUIRE(mxlFlowSynchronizationGroupAddReader(group, videoReader) == MXL_STATUS_OK);
    REQUIRE(mxlFlowSynchronizationGroupAddReader(group, dataReader) == MXL_STATUS_OK);

    auto const index = mxlTimestampToIndex(&GRAIN_RATE, mxlGetTime()) + 1;
    auto const timestamp = mxlIndexToTimestamp(&GRAIN_RATE, index);

    auto videoStatus = MXL_ERR_UNKNOWN;
    auto dataStatus = MXL_ERR_UNKNOWN;
    auto const start = std::chrono::steady_clock::now();
    auto dataThread = std::thread{[&]
        {
            dataStatus = commitGrainAfter(dataWriter, index, std::chrono::milliseconds{50});
        }};
    auto videoThread = std::thread{[&]
        {
            videoStatus = commitGrainSlotAfter(videoWriter, index, std::chrono::milliseconds{250});
        }};

    auto const waitStatus = mxlFlowSynchronizationGroupWaitForDataAt(group, timestamp, 2'000'000'000ULL);
    auto const waited = std::chrono::steady_clock::now() - start;

    videoThread.join();
    dataThread.join();

    REQUIRE(videoStatus == MXL_STATUS_OK);
    REQUIRE(dataStatus == MXL_STATUS_OK);
    REQUIRE(waitStatus == MXL_STATUS_OK);
    // The wait completed only after the slow backend member committed (250 ms, minus scheduling slack).
    CHECK(waited >= std::chrono::milliseconds{200});

    // A grain that nobody writes: the group must block until the timeout instead of reporting data.
    auto const missingTimestamp = mxlIndexToTimestamp(&GRAIN_RATE, index + 1000);
    auto const missingStart = std::chrono::steady_clock::now();
    auto const missingStatus = mxlFlowSynchronizationGroupWaitForDataAt(group, missingTimestamp, 200'000'000ULL);
    CHECK(missingStatus != MXL_STATUS_OK);
    CHECK(std::chrono::steady_clock::now() - missingStart >= std::chrono::milliseconds{150});

    REQUIRE(mxlReleaseFlowSynchronizationGroup(instance, group) == MXL_STATUS_OK);
    REQUIRE(mxlReleaseFlowReader(instance, videoReader) == MXL_STATUS_OK);
    REQUIRE(mxlReleaseFlowReader(instance, dataReader) == MXL_STATUS_OK);
    REQUIRE(mxlReleaseFlowWriter(instance, videoWriter) == MXL_STATUS_OK);
    REQUIRE(mxlReleaseFlowWriter(instance, dataWriter) == MXL_STATUS_OK);
    mxlDestroyInstance(instance);
}
