"""C-L3b 调参：内层切分不许越过切点，选中规则只认 AP。"""

from __future__ import annotations

import unittest
from unittest import mock

import pandas as pd
from bank import config as cfg_mod
from bank import timeaxis as T
from bank_training import tune


def _latest_tune_mapping() -> object:
    with mock.patch.object(cfg_mod, "load_config", wraps=cfg_mod.load_config):
        return cfg_mod.load_config()


def _train_frame() -> pd.DataFrame:
    dates = pd.to_datetime(["2008-11-10", "2009-03-31", "2009-04-01", "2009-09-30", "2009-10-01"])
    return pd.DataFrame({T.DATE: dates, "label": [1, 0, 0, 1, 0]})


class InnerSplitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = cfg_mod.load_config()

    def test_boundary_goes_to_holdout_and_cut_is_excluded(self) -> None:
        fit, holdout = tune.inner_split(_train_frame(), self.cfg)
        self.assertEqual(
            list(pd.to_datetime(fit[T.DATE]).dt.strftime("%Y-%m-%d")),
            ["2008-11-10", "2009-03-31"],
        )
        self.assertEqual(
            list(pd.to_datetime(holdout[T.DATE]).dt.strftime("%Y-%m-%d")),
            ["2009-04-01", "2009-09-30"],
        )

    def test_no_overlap_and_nothing_from_validation(self) -> None:
        fit, holdout = tune.inner_split(_train_frame(), self.cfg)
        self.assertEqual(len(fit) + len(holdout), 4)  # 切点当天那行不算
        self.assertLess(pd.to_datetime(holdout[T.DATE]).max(), pd.Timestamp(self.cfg.split.cut))

    def test_rejects_inner_cut_after_split_cut(self) -> None:
        with mock.patch.object(tune, "INNER_CUT", "2010-01-01"):
            with self.assertRaises(ValueError):
                tune.inner_split(_train_frame(), self.cfg)

    def test_rejects_empty_segment(self) -> None:
        early = pd.DataFrame({T.DATE: pd.to_datetime(["2008-11-10"]), "label": [1]})
        with self.assertRaises(ValueError):
            tune.inner_split(early, self.cfg)


class BestTest(unittest.TestCase):
    def test_picks_largest_ap(self) -> None:
        rows = [{"c": 0.3, "ap": 0.26}, {"c": 10.0, "ap": 0.29}, {"c": 1.0, "ap": 0.27}]
        self.assertEqual(tune.best(rows)["c"], 10.0)

    def test_empty_grid_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            tune.best([])


if __name__ == "__main__":
    unittest.main()
