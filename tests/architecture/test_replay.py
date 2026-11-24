"""守门测试：回放承诺不许注水。

回放的承诺是"每个 step-* tag 都能独立通过质量门"。这条承诺靠三件事成立：
1. tag 指向的提交里**真的有** `scripts/ci.py`（早于质量门的步骤不许打 tag）；
2. tag 都在当前历史里（是 HEAD 的祖先），不是悬空的孤立提交；
3. 每个 tag 在 `docs/steps/` 里有对应的三行说明（目标/验收/不做）。

这三条任何一条破了，"回放全绿"就变成表演。
"""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8"
    )


def step_tags() -> list[str]:
    return sorted(line.strip() for line in git("tag", "-l", "step-*").stdout.splitlines() if line)


def docs_root() -> Path:
    """步骤卡看**主工作区**，不看当前工作目录。

    回放时每个 tag 在独立 worktree 里跑质量门，那里的 `docs/steps/` 是那个 tag 当年的样子；
    步骤卡是后来回写的（见 `docs/steps/README.md`），所以"这张卡有没有"要在主工作区判断，
    否则回放越老越红，红的原因还跟代码无关。
    """
    common = git("rev-parse", "--path-format=absolute", "--git-common-dir")
    if common.returncode == 0 and common.stdout.strip():
        return Path(common.stdout.strip()).parent / "docs" / "steps"
    return ROOT / "docs" / "steps"


class ReplayableTagsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tags = step_tags()

    def test_has_tags(self) -> None:
        self.assertGreaterEqual(len(self.tags), 5, "step-* tag 太少，回放没有意义")

    def test_every_tag_contains_the_gate_script(self) -> None:
        for tag in self.tags:
            exists = git("cat-file", "-e", f"{tag}:scripts/ci.py")
            self.assertEqual(exists.returncode, 0, f"{tag} 里没有 scripts/ci.py，不该打 tag")

    def test_every_tag_is_ancestor_of_head(self) -> None:
        for tag in self.tags:
            merged = git("merge-base", "--is-ancestor", tag, "HEAD")
            self.assertEqual(merged.returncode, 0, f"{tag} 不在当前历史里")

    def test_every_tag_has_step_doc(self) -> None:
        root = docs_root()
        for tag in self.tags:
            name = tag.removeprefix("step-")
            self.assertTrue((root / f"{name}.md").is_file(), f"缺 docs/steps/{name}.md")


if __name__ == "__main__":
    unittest.main()
