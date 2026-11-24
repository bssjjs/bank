"""名额策略：把名额比例换成阈值（讲义 C-L1a 的 A 口径）。

手算基准（`n=10`、`capacity=0.2`、`medium_fraction=0.4`）：
- 名额边界取第 `ceil(10*0.2)=2` 名 → 「优先」档正好 2 人；
- 次优带取第 `ceil(10*0.4)=4` 名 → 「备选」是第 3、4 名；
- 阈值必须是**真实存在的分数**，不能是两个人之间插值出来的数；
- `分数 == 阈值` 算进这一档（`>=`），浮点比较不能把边界上的人丢掉。
"""

from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np
from bank import policy as P
from bank_training.export import _level_cutoffs

SCORES = np.asarray([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05], dtype="float64")


class FitTest(unittest.TestCase):
    def test_thresholds_are_real_scores_and_sit_on_rank(self) -> None:
        fitted = P.fit_capacity_quantile(SCORES, capacity=0.2, medium_fraction=0.4)
        self.assertIn(fitted.quota_threshold, SCORES)
        self.assertIn(fitted.watch_threshold, SCORES)
        self.assertEqual(fitted.quota_threshold, 0.8)
        self.assertEqual(fitted.watch_threshold, 0.6)

    def test_priority_band_is_exactly_the_quota(self) -> None:
        fitted = P.fit_capacity_quantile(SCORES, capacity=0.2, medium_fraction=0.4)
        self.assertEqual(int(fitted.selected(SCORES).sum()), 2)

    def test_three_bands_follow_option_a(self) -> None:
        fitted = P.fit_capacity_quantile(SCORES, capacity=0.2, medium_fraction=0.4)
        self.assertEqual(fitted.level(0.8), "优先")
        self.assertEqual(fitted.level(fitted.quota_threshold), "优先")
        self.assertEqual(fitted.level(0.7), "备选")
        self.assertEqual(fitted.level(fitted.watch_threshold), "备选")
        self.assertEqual(fitted.level(0.59), "不建议")
        self.assertEqual(
            fitted.level_of(SCORES),
            [
                "优先",
                "优先",
                "备选",
                "备选",
                "不建议",
                "不建议",
                "不建议",
                "不建议",
                "不建议",
                "不建议",
            ],
        )

    def test_old_rank_math_still_gives_the_quota_boundary(self) -> None:
        rng = np.random.default_rng(7)
        scores = rng.random(500)
        quota_n = 50
        ranked = np.sort(scores)[::-1]
        cutoffs, fitted = _level_cutoffs(
            scores, capacity=quota_n / len(scores), medium_fraction=0.4
        )
        self.assertEqual(cutoffs["优先"], float(ranked[quota_n - 1]))
        self.assertEqual(cutoffs["备选"], fitted.watch_threshold)
        self.assertLess(cutoffs["备选"], cutoffs["优先"])
        self.assertEqual(int(fitted.selected(scores).sum()), quota_n)

    def test_watch_band_ends_where_medium_fraction_says(self) -> None:
        rng = np.random.default_rng(11)
        scores = rng.random(1000)
        fitted = P.fit_capacity_quantile(scores, capacity=0.1, medium_fraction=0.4)
        covered = int((scores >= fitted.watch_threshold).sum())
        self.assertEqual(covered, math.ceil(1000 * 0.4))

    def test_ties_at_boundary_all_count_as_priority(self) -> None:
        tied = np.asarray([0.5] * 4 + [0.1] * 4, dtype="float64")
        fitted = P.fit_capacity_quantile(tied, capacity=0.25, medium_fraction=0.5)
        self.assertEqual(fitted.quota_threshold, 0.5)
        self.assertEqual(int(fitted.selected(tied).sum()), 4)

    def test_bad_scores_are_rejected(self) -> None:
        with self.assertRaises(P.PolicyError):
            P.fit_capacity_quantile(np.asarray([]), capacity=0.1, medium_fraction=0.4)
        with self.assertRaises(P.PolicyError):
            P.fit_capacity_quantile(np.asarray([0.5, np.nan]), capacity=0.1, medium_fraction=0.4)


class ParamTest(unittest.TestCase):
    def test_medium_fraction_must_exceed_capacity(self) -> None:
        with self.assertRaises(P.PolicyError):
            P.validate_params(0.1, 0.1)
        with self.assertRaises(P.PolicyError):
            P.validate_params(0.2, 0.1)
        P.validate_params(0.1, 0.4)

    def test_capacity_range(self) -> None:
        for bad in (0.0, 1.0, -0.1, 1.5):
            with self.assertRaises(P.PolicyError):
                P.validate_params(bad, 0.4)


class RoundTripTest(unittest.TestCase):
    def test_floats_survive_exactly(self) -> None:
        fitted = P.fit_capacity_quantile(SCORES, capacity=0.2, medium_fraction=0.4)
        back = P.DecisionPolicy.from_dict(fitted.to_dict())
        self.assertEqual(back, fitted)
        self.assertEqual(back.quota_threshold, fitted.quota_threshold)

    def test_no_rounding_happens(self) -> None:
        # n=3、capacity=0.5 → 名额 2 人，边界落在中间那个 13 位小数的分数上。
        # 四舍五入到 6 位会变成 0.123457：边界上的人就可能被换掉，讲义 C-L1a 要求存原始 float。
        scores = np.asarray([0.9, 0.1234567890123, 0.05], dtype="float64")
        fitted = P.fit_capacity_quantile(scores, capacity=0.5, medium_fraction=0.9)
        self.assertEqual(fitted.quota_threshold, 0.1234567890123)
        back = P.DecisionPolicy.from_dict(fitted.to_dict())
        self.assertEqual(back.quota_threshold, 0.1234567890123)

    def test_unknown_and_missing_keys_are_rejected(self) -> None:
        fitted = P.fit_capacity_quantile(SCORES, capacity=0.2, medium_fraction=0.4)
        data = fitted.to_dict()
        data["extra"] = 1
        with self.assertRaises(P.PolicyError):
            P.DecisionPolicy.from_dict(data)
        data = fitted.to_dict()
        del data["quota_threshold"]
        with self.assertRaises(P.PolicyError):
            P.DecisionPolicy.from_dict(data)

    def test_unknown_policy_name_is_rejected(self) -> None:
        self.assertEqual(P.POLICY_NAMES, ("capacity_quantile",))
        with self.assertRaises(P.PolicyError):
            P.fit_policy("top_k", SCORES, capacity=0.2, medium_fraction=0.4)


class ConfigTest(unittest.TestCase):
    def test_real_config(self) -> None:
        config = P.load_decision_config()
        self.assertEqual(config.schema_version, 1)
        self.assertEqual(config.decision.policy, "capacity_quantile")
        self.assertAlmostEqual(config.decision.medium_fraction, 0.40)

    def _write(self, body: str) -> Path:
        root = P.DECISION_CONFIG_PATH.parent.parent / ".tmp" / "policy-config"
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"decision-{abs(hash(body))}.yaml"
        path.write_text(body, encoding="utf-8", newline="\n")
        return path

    def test_typo_key_is_rejected(self) -> None:
        path = self._write(
            "schema_version: 1\ndecision:\n  policy: capacity_quantile\n  medium_fracton: 0.4\n"
        )
        with self.assertRaises(P.PolicyError):
            P.load_decision_config(path)

    def test_unknown_policy_is_rejected(self) -> None:
        path = self._write(
            "schema_version: 1\ndecision:\n  policy: top_k\n  medium_fraction: 0.4\n"
        )
        with self.assertRaises(P.PolicyError):
            P.load_decision_config(path)

    def test_medium_fraction_out_of_range_is_rejected(self) -> None:
        for bad in ("0", "1", "1.2"):
            path = self._write(
                f"schema_version: 1\ndecision:\n  policy: capacity_quantile\n"
                f"  medium_fraction: {bad}\n"
            )
            with self.assertRaises(P.PolicyError):
                P.load_decision_config(path)

    def test_version_two_is_rejected(self) -> None:
        path = self._write(
            "schema_version: 2\ndecision:\n  policy: capacity_quantile\n  medium_fraction: 0.4\n"
        )
        with self.assertRaises(P.PolicyError):
            P.load_decision_config(path)

    def test_missing_file_is_rejected(self) -> None:
        with self.assertRaises(P.PolicyError):
            P.load_decision_config(Path("没有这个文件.yaml"))


if __name__ == "__main__":
    unittest.main()
