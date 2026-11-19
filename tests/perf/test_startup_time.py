#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
# SPDX-License-Identifier: Apache-2.0
"""startup_time.py analizės testai (planas production-ready §3.3, G5): gryna logika, GPU nereikia."""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import startup_time as st  # noqa: E402


def line(seq, oks, event=None):
    if event:
        return '{"v":1,"event":"%s","t":"x","tai":1}' % event
    flows = ",".join('{"n":%d,"ok":%d}' % (i, ok) for i, ok in enumerate(oks))
    return '{"v":1,"t":"x","tai":%d,"seq":%d,"flows":[%s]}' % (seq, seq, flows)


class FirstOk(unittest.TestCase):
    def test_first_line_with_any_ok(self):
        self.assertEqual(st.first_ok([line(1, [0, 0]), line(2, [0, 3]), line(3, [5, 5])]), 2)

    def test_none_when_no_ok(self):
        self.assertIsNone(st.first_ok([line(1, [0]), line(2, [0])]))

    def test_all_flows_mode(self):
        self.assertEqual(st.first_ok([line(1, [0, 3]), line(2, [4, 0]), line(3, [1, 1])], all_flows=True), 3)

    def test_ignores_events_partial_and_garbage(self):
        lines = [line(0, [], event="InputAttached"), "{\"v\":1,\"flo", "", line(7, [2])]
        self.assertEqual(st.first_ok(lines), 7)


class Summary(unittest.TestCase):
    def test_summary_p50_max(self):
        self.assertEqual(st.summary([0.3, 0.1, 0.2, None]), {"n": 3, "failed": 1, "p50_s": 0.2, "max_s": 0.3})

    def test_summary_empty(self):
        self.assertEqual(st.summary([]), {"n": 0, "failed": 0, "p50_s": None, "max_s": None})


if __name__ == "__main__":
    unittest.main()
