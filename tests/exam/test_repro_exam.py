"""考卷：指纹与规范化（A2）。人出题，智能体只负责考过。

题目考"同一份数据换了机器还算不算同一份"：
- 文件指纹按 1 MiB 分块读，大文件与小文件都要和 hashlib 一致；
- 规范化 JSON 只认值，不认键序，也不许把中文转义成 \\uXXXX；
- 表指纹对行序、列序敏感（数据集里行序就是时间序，不能丢）；
- 浮点用 %.10g，所以 0.1 与 0.1000000000001 这类"第 11 位才不同"的值会被视为同一个数
  ——这是**有意的口径**，题里要求它成立，答辩时说清代价。
"""

from __future__ import annotations

import atexit
import hashlib
import shutil
import unittest
import uuid
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from bank.repro import (
    canonical_json,
    git_dirty,
    git_sha,
    sha256_file,
    sha256_json,
    sha256_records,
    sha256_table,
    sha256_text,
)

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = ROOT / ".tmp" / "exam"

ABC = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
EMPTY = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

_cleanup_registered = False


def scratch_file(name: str, data: bytes) -> Path:
    """造一个临时文件，进程退出时连同父目录一起删掉。

    「大文件按 1 MiB 分块」那道题每次会写一个 1 MB 的 big.bin；早先版本写完不删，
    跑几百次测试就把 .tmp/exam 撑到 139 MB。这里注册一次 atexit，别让它再漏。
    """
    global _cleanup_registered
    SCRATCH.mkdir(parents=True, exist_ok=True)
    if not _cleanup_registered:
        atexit.register(shutil.rmtree, SCRATCH, True)
        _cleanup_registered = True
    path = SCRATCH / f"{uuid.uuid4().hex}-{name}"
    path.write_bytes(data)
    return path


class FileDigestTest(unittest.TestCase):
    def test_known_vector(self) -> None:
        self.assertEqual(sha256_text("abc"), ABC)

    def test_empty_file(self) -> None:
        self.assertEqual(sha256_file(scratch_file("empty.bin", b"")), EMPTY)

    def test_big_file_is_read_in_chunks_but_hashes_whole(self) -> None:
        # 比 1 MiB 大一点点：跨块边界最容易写错
        data = bytes(range(256)) * 4097
        self.assertGreater(len(data), 1 << 20)
        self.assertEqual(
            sha256_file(scratch_file("big.bin", data)), hashlib.sha256(data).hexdigest()
        )

    def test_utf8_text_is_hashed_as_utf8(self) -> None:
        self.assertEqual(sha256_text("认购"), hashlib.sha256("认购".encode()).hexdigest())


class CanonicalJsonTest(unittest.TestCase):
    def test_key_order_does_not_matter(self) -> None:
        self.assertEqual(canonical_json({"a": 1, "b": 2}), canonical_json({"b": 2, "a": 1}))

    def test_chinese_is_not_escaped(self) -> None:
        self.assertIn("认购", canonical_json({"target": "认购"}))
        self.assertNotIn("\\u", canonical_json({"target": "认购"}))

    def test_no_whitespace(self) -> None:
        self.assertEqual(canonical_json({"a": [1, 2]}), '{"a":[1,2]}')

    def test_path_and_dates_become_stable_strings(self) -> None:
        payload = {
            "p": Path("data") / "frozen",
            "d": date(2009, 10, 1),
            "t": datetime(2010, 11, 17, 3, 4),
        }
        text = canonical_json(payload)
        self.assertIn('"p":"data/frozen"', text)
        self.assertIn('"d":"2009-10-01"', text)
        self.assertIn('"t":"2010-11-17T03:04:00"', text)

    def test_unknown_type_is_not_silently_stringified(self) -> None:
        with self.assertRaises(TypeError) as caught:
            canonical_json({"s": {1, 2}})
        self.assertIn("set", str(caught.exception))

    def test_sha256_json_is_seeded_by_canonical_json(self) -> None:
        payload = {"b": [1, 2, 3], "a": "认购"}
        self.assertEqual(sha256_json(payload), sha256_text(canonical_json(payload)))

    def test_records_ignore_key_order_inside_each_record(self) -> None:
        left = [{"a": 1, "b": 2}]
        right = [{"b": 2, "a": 1}]
        self.assertEqual(sha256_records(left), sha256_records(right))


class TableDigestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pd.DataFrame(
            {"年龄": [46, 31], "余额": [81204, -1200], "职业": ["退休", "蓝领工人"]}
        )

    def test_same_table_same_digest(self) -> None:
        self.assertEqual(sha256_table(self.df), sha256_table(self.df.copy()))

    def test_column_order_matters(self) -> None:
        shuffled = self.df[["余额", "年龄", "职业"]]
        self.assertNotEqual(sha256_table(self.df), sha256_table(shuffled))

    def test_row_order_matters(self) -> None:
        reversed_rows = self.df.iloc[::-1].reset_index(drop=True)
        self.assertNotEqual(sha256_table(self.df), sha256_table(reversed_rows))

    def test_value_change_changes_digest(self) -> None:
        changed = self.df.copy()
        changed.loc[0, "年龄"] = 47
        self.assertNotEqual(sha256_table(self.df), sha256_table(changed))

    def test_float_is_formatted_to_ten_significant_digits(self) -> None:
        # 第 11 位之后不同，肉眼与业务都无所谓；口径上算同一个数（文档里写了这条取舍）
        left = pd.DataFrame({"x": [0.1]})
        right = pd.DataFrame({"x": [0.1000000000001]})
        self.assertEqual(sha256_table(left), sha256_table(right))


class GitStateTest(unittest.TestCase):
    def test_git_sha_is_a_commit_id(self) -> None:
        sha = git_sha()
        self.assertIsNotNone(sha, "在仓库里跑测试却拿不到 git sha")
        assert sha is not None
        self.assertEqual(len(sha), 40)
        self.assertTrue(all(c in "0123456789abcdef" for c in sha.lower()))

    def test_git_dirty_is_a_bool(self) -> None:
        self.assertIsInstance(git_dirty(), bool)


if __name__ == "__main__":
    unittest.main()
