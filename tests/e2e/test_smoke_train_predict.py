"""M8 端到端冒烟：合成小数据 → 派生特征 → 训练 → 预测。

不读原始库、不读冻结集，也不写模型包；只验证训练与推理这两段能接上。
"""

from __future__ import annotations

import unittest

import numpy as np
from bank import catalog
from bank import dataset as D
from bank import schema as S
from bank_training.pipeline import build_model

from tests.support import synthetic_raw_frame


class SmokeTrainPredictTest(unittest.TestCase):
    def test_synthetic_data_can_train_and_predict(self) -> None:
        raw = synthetic_raw_frame(rows=300, seed=11)
        data = D.build_dataset(raw)
        features = D.model_features("B")
        split_at = 240

        model = build_model(
            "lr",
            catalog.load_catalog(),
            features,
            c=0.5,
            max_iter=500,
        )
        model.fit(data[features].iloc[:split_at], data[S.TARGET].iloc[:split_at])
        scores = model.predict_proba(data[features].iloc[split_at:])[:, 1]

        self.assertEqual(len(scores), len(data) - split_at)
        self.assertTrue(np.isfinite(scores).all())
        self.assertTrue(((scores >= 0.0) & (scores <= 1.0)).all())
        self.assertTrue(set(model.predict(data[features].iloc[split_at:])) <= {0, 1})


if __name__ == "__main__":
    unittest.main()
