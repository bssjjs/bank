"""守门测试：基线是只读参照，不许被悄悄改。

三件事：
1. `scripts/freeze_baseline.py --check` 必须通过——基线文件与封条一致；
2. 封条自身的格式（排序、正斜杠、64 位十六进制、两空格分隔、LF 结尾）；
3. 基线里的数字必须与文档里的数字一致（模型卡那份表就是从这些 run 抄出来的）。
   第 3 条防的是"改了模型、忘了改文档"，或者反过来"文档写了个没跑出来的数"。
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import unittest
from pathlib import Path

from bank.repro import sha256_file

ROOT = Path(__file__).resolve().parents[2]
SUMS = ROOT / "baselines" / "SHA256SUMS"
V1 = ROOT / "baselines" / "v1"
MAIN_RUN = V1 / "runs" / "m3-lr-主模型.json"
MODEL_CARD = ROOT / "docs" / "model_card.md"

LINE = re.compile(r"^(?P<digest>[0-9a-f]{64})  (?P<path>\S+)$")


def parse_line(line: str) -> tuple[str, str]:
    """拆一行封条，返回 (digest, path)；格式不对直接失败，别让后续断言在 None 上继续。"""
    match = LINE.match(line)
    if match is None:
        raise AssertionError(f"封条行格式不对：{line!r}")
    return match.group("digest"), match.group("path")


class BaselineCheckTest(unittest.TestCase):
    def test_freeze_baseline_check_passes(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/freeze_baseline.py", "--check"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        self.assertEqual(
            result.returncode, 0, f"--check 未通过：\n{result.stdout}\n{result.stderr}"
        )
        self.assertIn("基线校验通过", result.stdout)


class SealsFormatTest(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = SUMS.read_bytes()
        self.text = self.raw.decode("utf-8")
        self.lines = self.text.splitlines()

    def test_ends_with_single_lf(self) -> None:
        self.assertTrue(self.raw.endswith(b"\n"))
        self.assertFalse(self.raw.endswith(b"\n\n"))

    def test_every_line_matches_format(self) -> None:
        for line in self.lines:
            parse_line(line)

    def test_paths_sorted_and_forward_slash(self) -> None:
        paths = [parse_line(line)[1] for line in self.lines]
        self.assertEqual(paths, sorted(paths))
        for path in paths:
            self.assertNotIn("\\", path, "封条里必须是正斜杠，换机器才可比")
            self.assertTrue(path.startswith("baselines/"))

    def test_digests_match_files(self) -> None:
        for line in self.lines:
            digest, path = parse_line(line)
            self.assertEqual(sha256_file(ROOT / path), digest, path)


class NumbersAgreeTest(unittest.TestCase):
    def test_model_card_numbers_come_from_the_main_run(self) -> None:
        metrics = json.loads(MAIN_RUN.read_text(encoding="utf-8"))["metrics"]
        card = MODEL_CARD.read_text(encoding="utf-8")
        # 模型卡里保留四位小数；这里按同样的精度比对，口径是"文档数字必须来自这次运行"
        for key, digits in (
            ("ap", 4),
            ("roc_auc", 4),
            ("precision_at_quota", 4),
            ("recall_at_quota", 4),
            ("brier", 4),
        ):
            value = f"{metrics[key]:.{digits}f}"
            self.assertIn(value, card, f"模型卡里找不到 {key}={value}")

    def test_frozen_baseline_matches_current_frozen_ledger(self) -> None:
        snapshot = json.loads((V1 / "frozen.json").read_text(encoding="utf-8"))
        current = json.loads((ROOT / "data" / "frozen" / "FROZEN.json").read_text(encoding="utf-8"))
        self.assertEqual(snapshot["tables"], current["tables"])

    def test_baseline_records_quota_and_rows(self) -> None:
        main = json.loads(MAIN_RUN.read_text(encoding="utf-8"))
        self.assertEqual(main["params"]["quota_n"], 356)
        self.assertEqual(main["params"]["n_valid"], 3556)
        self.assertEqual(main["params"]["positives_valid"], 1782)

    def test_readme_parity_table_matches_the_baseline(self) -> None:
        """README 的算法一致性表必须与 `baselines/v1/algorithm_parity.json` 逐格相等。

        这张表是手抄进 README 的，抄错一位就是「报告数字与命令输出对不上」——
        任务书的一票否决项。所以让测试盯住它，而不是靠人每次复核。
        """
        reference = json.loads((V1 / "algorithm_parity.json").read_text(encoding="utf-8"))
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        counts = {"lr": 264, "rf": 272}
        for name, row in reference["algorithms"].items():
            line = next(
                (item for item in readme.splitlines() if item.startswith(f"| `{name}`")),
                None,
            )
            self.assertIsNotNone(line, f"README 里找不到 `{name}` 这一行")
            cells = [cell.strip() for cell in str(line).strip("|").split("|")]
            self.assertEqual(len(cells), 6, f"`{name}` 这一行的列数不对：{cells}")
            self.assertEqual(cells[1], f"{row['ap']:.6f}", name)
            self.assertEqual(cells[2], f"{row['roc_auc']:.6f}", name)
            self.assertEqual(cells[3], f"{row['precision_at_quota']:.6f}", name)
            self.assertEqual(cells[4], f"{row['recall_at_quota']:.6f}", name)
            self.assertEqual(cells[5], str(counts[name]), name)


if __name__ == "__main__":
    unittest.main()
