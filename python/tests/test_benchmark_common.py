"""Tests for the per-solver result files of benchmark_mm.py / benchmark_mpc.py
(replace_rows, check_header, load_solver_tables in benchmark_common.py).

Run from the repository root with:
    python3 -m unittest discover -s python/tests -t python
"""

import csv
import tempfile
import unittest
from pathlib import Path

from benchmark_common import check_header, load_solver_tables, replace_rows


def _read(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


class ReplaceRows(unittest.TestCase):
    FIELDS = ["name", "t"]

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "mm_kspqp.csv"

    def tearDown(self):
        self._tmp.cleanup()

    def test_creates_the_file_with_its_header(self):
        replace_rows(self.path, [{"name": "A", "t": 1.0}], ["name"], self.FIELDS)
        with open(self.path, newline="") as fh:
            self.assertEqual(next(csv.reader(fh)), self.FIELDS)
        self.assertEqual(_read(self.path), [{"name": "A", "t": "1.0"}])

    def test_rerun_replaces_matching_rows_in_place_and_keeps_the_rest(self):
        for name, t in [("A", 1), ("B", 2), ("C", 3)]:
            replace_rows(self.path, [{"name": name, "t": t}], ["name"], self.FIELDS)
        replace_rows(self.path, [{"name": "B", "t": 20}], ["name"], self.FIELDS)
        replace_rows(self.path, [{"name": "D", "t": 4}], ["name"], self.FIELDS)
        self.assertEqual([(r["name"], r["t"]) for r in _read(self.path)],
                         [("A", "1"), ("B", "20"), ("C", "3"), ("D", "4")])

    def test_a_key_covering_several_rows_replaces_all_of_them(self):
        # MPC: the key is the configuration, which covers every step and repeat.
        fields = ["M", "N", "step", "rep", "t"]
        old = [{"M": 5, "N": 20, "step": s, "rep": r, "t": 1} for s in range(2) for r in range(2)]
        other = [{"M": 10, "N": 20, "step": 0, "rep": 0, "t": 7}]
        replace_rows(self.path, old + other, ["M", "N"], fields)
        replace_rows(self.path, [{"M": 5, "N": 20, "step": 0, "rep": 0, "t": 2}], ["M", "N"], fields)
        rows = _read(self.path)
        self.assertEqual([(r["M"], r["step"], r["rep"], r["t"]) for r in rows],
                         [("5", "0", "0", "2"), ("10", "0", "0", "7")])

    def test_refuses_a_file_with_another_header(self):
        with open(self.path, "w", newline="") as fh:
            csv.writer(fh).writerows([["name", "old_column"], ["A", "1"]])
        with self.assertRaises(SystemExit):
            replace_rows(self.path, [{"name": "A", "t": 1}], ["name"], self.FIELDS)
        with self.assertRaises(SystemExit):
            check_header(self.path, self.FIELDS)
        check_header(self.path, ["name", "old_column"])  # its own header passes
        self.assertEqual(_read(self.path), [{"name": "A", "old_column": "1"}])


class LoadSolverTables(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self.ksp, self.qpalm = tmp / "mm_kspqp.csv", tmp / "mm_qpalm.csv"
        replace_rows(self.ksp, [{"name": n, "ssn_time": t, "host": "a"} for n, t in [("A", 1), ("B", 2)]],
                     ["name"], ["name", "ssn_time", "host"])
        replace_rows(self.qpalm, [{"name": n, "qpalm_time": t, "host": "b"} for n, t in [("B", 5), ("C", 6)]],
                     ["name"], ["name", "qpalm_time", "host"])

    def tearDown(self):
        self._tmp.cleanup()

    def test_inner_join_keeps_the_common_problems(self):
        df = load_solver_tables({"ksp-qp": self.ksp, "qpalm": self.qpalm}, ["name"])
        self.assertEqual(df["name"].tolist(), ["B"])
        self.assertEqual(df["ssn_time"].tolist(), [2])
        self.assertEqual(df["qpalm_time"].tolist(), [5])

    def test_outer_join_keeps_all_and_suffixes_shared_columns(self):
        df = load_solver_tables({"ksp-qp": self.ksp, "qpalm": self.qpalm}, ["name"], how="outer")
        self.assertEqual(sorted(df["name"].tolist()), ["A", "B", "C"])
        self.assertIn("host_ksp-qp", df.columns)
        self.assertIn("host_qpalm", df.columns)
        self.assertNotIn("host", df.columns)


if __name__ == "__main__":
    unittest.main()
