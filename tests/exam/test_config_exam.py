"""考卷：配置加载（A3）。人出题，智能体只负责"考过"，不许改题目。

题目考的是"口径数字写错会怎样"：
- 拼错键、重复键、越界值，必须当场报错并指出哪个键；
- 配置是冻结的，运行时改不了；
- 指纹只认值，不认注释和缩进；
- 名额人数只由验证集行数与 frac 决定。

这些是答辩会被问到的点（"缺一列请求在哪层拦住""名额人数怎么算"），
所以答案必须是代码里真的会发生的，不是文档里写得好听的。
"""

from __future__ import annotations

import unittest
import uuid
from pathlib import Path

from bank.config import CONFIG_PATH, ConfigError, config_fingerprint, load_config, quota_size
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[2]
BASE = CONFIG_PATH.read_text(encoding="utf-8")
SCRATCH = ROOT / ".tmp" / "exam"


def variant(*replacements: tuple[str, str], extra: str = "") -> Path:
    """把 base.yaml 改几处写成临时文件；replacements 里 (旧, 新) 必须真的存在。"""
    text = BASE
    for old, new in replacements:
        if old not in text:
            raise AssertionError(f"base.yaml 里找不到要替换的片段：{old!r}")
        text = text.replace(old, new, 1)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    path = SCRATCH / f"{uuid.uuid4().hex}.yaml"
    path.write_text(text + extra, encoding="utf-8", newline="\n")
    return path


class LoadTest(unittest.TestCase):
    def test_real_config_loads(self) -> None:
        cfg = load_config()
        self.assertEqual(cfg.schema_version, 1)
        self.assertEqual(cfg.features.set, "B")
        self.assertEqual(cfg.split.buffer_days, 0)
        self.assertEqual(cfg.target.window_days, 0)

    def test_missing_file_is_an_error(self) -> None:
        with self.assertRaises(ConfigError):
            load_config(SCRATCH / "不存在的配置.yaml")

    def test_fingerprint_is_64_hex(self) -> None:
        digest = config_fingerprint(load_config())
        self.assertEqual(len(digest), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in digest))


class StrictnessTest(unittest.TestCase):
    def test_misspelled_key_is_rejected_with_its_name(self) -> None:
        # 多写一个不存在的键：拼错键必须当场报错，而不是被默默忽略
        path = variant(("  cut: 2009-10-01", "  cut: 2009-10-01\n  cutt: 2009-10-01"))
        with self.assertRaises(ConfigError) as caught:
            load_config(path)
        self.assertIn("cutt", str(caught.exception))

    def test_duplicate_key_is_rejected(self) -> None:
        path = variant(("seed: 20261122", "seed: 20261122\nseed: 1"))
        with self.assertRaises(ConfigError) as caught:
            load_config(path)
        self.assertIn("键重复", str(caught.exception))

    def test_bad_date_is_rejected(self) -> None:
        path = variant(("cut: 2009-10-01", "cut: 2009-13-01"))
        with self.assertRaises(ConfigError):
            load_config(path)

    def test_frozen_config_cannot_be_mutated(self) -> None:
        cfg = load_config()
        with self.assertRaises(ValidationError):
            setattr(cfg, "seed", 1)


class RangesTest(unittest.TestCase):
    def test_start_must_be_before_end(self) -> None:
        path = variant(("start: 2008-11-01", "start: 2011-12-31"))
        with self.assertRaises(ConfigError) as caught:
            load_config(path)
        self.assertIn("window.start", str(caught.exception))

    def test_cut_must_be_inside_window(self) -> None:
        path = variant(("cut: 2009-10-01", "cut: 2011-01-01"))
        with self.assertRaises(ConfigError) as caught:
            load_config(path)
        self.assertIn("split.cut", str(caught.exception))

    def test_buffer_days_upper_bound(self) -> None:
        path = variant(("buffer_days: 0", "buffer_days: 366"))
        with self.assertRaises(ConfigError) as caught:
            load_config(path)
        self.assertIn("buffer_days", str(caught.exception))

    def test_quota_frac_must_not_exceed_one(self) -> None:
        path = variant(("frac: 0.10", "frac: 1.01"))
        with self.assertRaises(ConfigError) as caught:
            load_config(path)
        self.assertIn("quota.frac", str(caught.exception))

    def test_quota_frac_one_is_allowed(self) -> None:
        cfg = load_config(variant(("frac: 0.10", "frac: 1.0")))
        self.assertEqual(cfg.quota.frac, 1.0)

    def test_label_window_must_stay_zero(self) -> None:
        path = variant(("window_days: 0", "window_days: 90"))
        with self.assertRaises(ConfigError) as caught:
            load_config(path)
        self.assertIn("window_days", str(caught.exception))

    def test_raw_table_must_match_schema(self) -> None:
        path = variant(("raw_table: 银行营销数据", "raw_table: 别的表"))
        with self.assertRaises(ConfigError) as caught:
            load_config(path)
        self.assertIn("raw_table", str(caught.exception))


class FingerprintTest(unittest.TestCase):
    def test_comments_and_blank_lines_do_not_change_fingerprint(self) -> None:
        base = config_fingerprint(load_config())
        same = config_fingerprint(load_config(variant(extra="\n# 只是加一行注释\n\n")))
        self.assertEqual(base, same)

    def test_value_change_changes_fingerprint(self) -> None:
        base = config_fingerprint(load_config())
        other = config_fingerprint(load_config(variant(("seed: 20261122", "seed: 20261123"))))
        self.assertNotEqual(base, other)


class QuotaTest(unittest.TestCase):
    def test_quota_matches_validation_rows(self) -> None:
        cfg = load_config()
        # 3,556 × 0.10 = 355.6 → 向上取整 356：名额按"人"算，不能四舍五入少一个人
        self.assertEqual(quota_size(3556, cfg), 356)
        self.assertEqual(quota_size(10, cfg), 1)

    def test_empty_validation_set_is_an_error(self) -> None:
        # 分母是 0 时不该给出「0 个名额」这种答案，而是直接拒绝
        with self.assertRaises(ValueError):
            quota_size(0, load_config())


if __name__ == "__main__":
    unittest.main()
