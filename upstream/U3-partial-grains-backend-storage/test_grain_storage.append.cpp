
///
/// Partial grains must work with a payload backend exactly as with host storage.
///
/// A writer commits a grain slice batch by slice batch through the slot descriptor API. After each commit a reader that
/// asks for at most the committed number of slices gets the grain with that validSlices value and the committed bytes,
/// and a reader that asks for one slice more gets MXL_ERR_OUT_OF_RANGE_TOO_EARLY. Reopening the incomplete grain keeps
/// its progress. Low latency media functions depend on this: they start processing the first slices of a grain before
/// the last ones are written.
///
TEST_CASE_PERSISTENT_FIXTURE(mxl::tests::mxlDomainFixture, "Grain storage : Partial grains with a payload backend",
    "[mxl flows][grain storage]")
{
    auto const flows = FlowPair{domain, "data/v210_flow.json", V210_FLOW_ID, TEST_BACKEND_OPTIONS};

    auto layout = mxlGrainStorageLayout{};
    REQUIRE(mxlFlowReaderGetStorageLayout(flows.reader(), &layout) == MXL_STATUS_OK);
    auto writerSlots = std::vector<mxlGrainStorage>(layout.slotCount);
    REQUIRE(mxlFlowWriterMapSlots(flows.writer(), layout.slotCount, writerSlots.data()) == MXL_STATUS_OK);
    auto readerSlots = std::vector<mxlGrainStorage>(layout.slotCount);
    REQUIRE(mxlFlowReaderMapSlots(flows.reader(), layout.slotCount, readerSlots.data()) == MXL_STATUS_OK);

    auto const rate = flows.configInfo().common.grainRate;
    auto const index = mxlGetCurrentIndex(&rate);
    auto grainInfo = mxlGrainInfo{};
    auto writerSlot = std::uint32_t{0};
    REQUIRE(mxlFlowWriterOpenGrainSlot(flows.writer(), index, &grainInfo, &writerSlot) == MXL_STATUS_OK);
    REQUIRE(grainInfo.validSlices == 0U);
    REQUIRE(grainInfo.totalSlices > 4U);

    auto const sliceSize = std::size_t{flows.configInfo().discrete.sliceSizes[0]};
    auto* const writerBytes = static_cast<std::uint8_t*>(writerSlots[writerSlot].host.pointer);
    auto const batch = static_cast<std::uint16_t>(grainInfo.totalSlices / 4U);

    for (auto committed = std::uint16_t{0}; committed < grainInfo.totalSlices;)
    {
        auto const next = static_cast<std::uint16_t>(std::min<unsigned>(committed + batch, grainInfo.totalSlices));
        // Each slice carries a byte that identifies it, so the reader can check that it sees exactly the committed slices.
        for (auto slice = committed; slice < next; ++slice)
        {
            std::memset(writerBytes + slice * sliceSize, static_cast<int>(slice & 0xFFU), sliceSize);
        }
        grainInfo.validSlices = next;
        REQUIRE(mxlFlowWriterCommitGrain(flows.writer(), &grainInfo) == MXL_STATUS_OK);
        committed = next;

        auto readInfo = mxlGrainInfo{};
        auto readerSlot = std::uint32_t{0};
        REQUIRE(mxlFlowReaderGetGrainSlot(flows.reader(), index, committed, 0U, &readInfo, &readerSlot) == MXL_STATUS_OK);
        REQUIRE(readerSlot == writerSlot);
        REQUIRE(readInfo.validSlices == committed);
        auto const* const readerBytes = static_cast<std::uint8_t const*>(readerSlots[readerSlot].host.pointer);
        REQUIRE(readerBytes[(committed - 1U) * sliceSize] == static_cast<std::uint8_t>((committed - 1U) & 0xFFU));

        if (committed < grainInfo.totalSlices)
        {
            // One slice more than committed is not available yet.
            REQUIRE(mxlFlowReaderGetGrainSlot(flows.reader(), index, static_cast<std::uint16_t>(committed + 1U), 0U, &readInfo,
                        &readerSlot) == MXL_ERR_OUT_OF_RANGE_TOO_EARLY);

            // Reopening the incomplete grain keeps its progress, as with host storage.
            auto reopened = mxlGrainInfo{};
            auto reopenedSlot = std::uint32_t{0};
            REQUIRE(mxlFlowWriterOpenGrainSlot(flows.writer(), index, &reopened, &reopenedSlot) == MXL_STATUS_OK);
            REQUIRE(reopenedSlot == writerSlot);
            REQUIRE(reopened.validSlices == committed);
            grainInfo = reopened;
        }
    }

    // The complete grain is returned for MXL_GRAIN_VALID_SLICES_ALL.
    auto readInfo = mxlGrainInfo{};
    auto readerSlot = std::uint32_t{0};
    REQUIRE(mxlFlowReaderGetGrainSlot(flows.reader(), index, MXL_GRAIN_VALID_SLICES_ALL, 0U, &readInfo, &readerSlot) == MXL_STATUS_OK);
    REQUIRE(readInfo.validSlices == readInfo.totalSlices);
}
