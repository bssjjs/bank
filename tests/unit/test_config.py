"""配置层：拼错的键必须立刻报错，分母口径必须算得对。"""

from __future__ import annotations

import unittest
from pathlib import Path

from bank import config as CFG

TMP = Path(__file__).resolve().parents[2] / ".tmp"


def write_yaml(name: str, text: str) -> Path:
    TMP.mkdir(exist_ok=True)
    path = TMP / name
    path.write_text(text, encoding="utf-8")
    return path


GOOD = """
schema_version: 1
seed: 20261122
data:
  raw_db: data/bank-marketing.db
  raw_table: 银行营销数据
  frozen_dir: data/frozen
window:
  start: 2008-11-01
  end: 2010-11-17
split:
  cut: 2009-10-01
  buffer_days: 0
quota:
  frac: 0.10
features:
  set: A
target:
  column: 是否认购
  positive: 是
  negative: 否
  window_days: 0
"""


class ConfigLoadTest(unittest.TestCase):
    def test_real_config_loads(self) -> None:
        cfg = CFG.load_config()
        self.assertEqual(cfg.window.start.isoformat(), "2008-11-01")
        self.assertEqual(cfg.window.end.isoformat(), "2010-11-17")
        self.assertEqual(cfg.split.cut.isoformat(), "2009-10-01")
        self.assertEqual(cfg.split.buffer_days, 0)
        self.assertEqual(cfg.quota.frac, 0.10)
        # E2 之后主口径改用特征集 B（去掉月份三列），证据在 ADR 0005
        self.assertEqual(cfg.features.set, "B")
        self.assertEqual(cfg.seed, 20261122)
        self.assertEqual(cfg.target.window_days, 0)

    def test_fingerprint_stable(self) -> None:
        self.assertEqual(
            CFG.config_fingerprint(CFG.load_config()), CFG.config_fingerprint(CFG.load_config())
        )

    def test_unknown_key_rejected(self) -> None:
        path = write_yaml(
            "bad-unknown.yaml", GOOD.replace("frac: 0.10", "frac: 0.10\n  fracs: 0.2")
        )
        with self.assertRaises(CFG.ConfigError) as ctx:
            CFG.load_config(path)
        self.assertIn("fracs", str(ctx.exception))

    def test_duplicate_key_rejected(self) -> None:
        path = write_yaml("bad-dup.yaml", GOOD + "\nseed: 1\n")
        with self.assertRaises(CFG.ConfigError) as ctx:
            CFG.load_config(path)
        self.assertIn("键重复", str(ctx.exception))

    def test_cut_outside_window_rejected(self) -> None:
        path = write_yaml("bad-cut.yaml", GOOD.replace("cut: 2009-10-01", "cut: 2008-01-01"))
        with self.assertRaises(CFG.ConfigError) as ctx:
            CFG.load_config(path)
        self.assertIn("cut", str(ctx.exception))

    def test_start_after_end_rejected(self) -> None:
        path = write_yaml(
            "bad-order.yaml",
            GOOD.replace("start: 2008-11-01", "start: 2011-01-01"),
        )
        with self.assertRaises(CFG.ConfigError) as ctx:
            CFG.load_config(path)
        self.assertIn("start", str(ctx.exception))

    def test_target_must_match_schema(self) -> None:
        path = write_yaml("bad-target.yaml", GOOD.replace("column: 是否认购", "column: 认购"))
        with self.assertRaises(CFG.ConfigError) as ctx:
            CFG.load_config(path)
        self.assertIn("column", str(ctx.exception))

    def test_missing_file_rejected(self) -> None:
        with self.assertRaises(CFG.ConfigError):
            CFG.load_config(TMP / "not-there.yaml")


class QuotaTest(unittest.TestCase):
    def test_verification_quota_is_356(self) -> None:
        cfg = CFG.load_config()
        self.assertEqual(CFG.quota_size(3556, cfg), 356)

    def test_tiny_set_rounds_up(self) -> None:
        cfg = CFG.load_config()
        self.assertEqual(CFG.quota_size(10, cfg), 1)

    def test_empty_validation_set_rejected(self) -> None:
        cfg = CFG.load_config()
        with self.assertRaises(ValueError):
            CFG.quota_size(0, cfg)


if __name__ == "__main__":
    unittest.main()
