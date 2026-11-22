"""规则基线：不用训练，凭业务常识打分。

它的作用不是"赢"，是给模型一个参照：如果逻辑回归连这套手写规则都赢不了，
那模型就没理由上线。规则只用联系当时就能拿到的信息，不许碰 `通话时长`。
"""

from __future__ import annotations

import pandas as pd
from bank import schema as S

# 每条的权重与理由（答辩会被问"凭什么给这个分"）：
# 3 分：上次营销成功 —— 表里认购率最高的一档，约占 3.3% 的行，命中率远超平均；
# 1 分：被联系过 —— 有历史记录说明是既有客户，比从未联系过的陌生号好打；
# 1 分：联系方式=手机 —— 手机可达率高于固话和"未知"；
# 1 分：教育程度=高等教育 —— 表内认购率随教育程度单调上升；
# 1 分：本次联系次数<=2 —— 联系次数越多越说明前面没谈成，是反向信号。
WEIGHTS = {
    "上次成功": 3.0,
    "被联系过": 1.0,
    "手机": 1.0,
    "高等教育": 1.0,
    "联系次数少": 1.0,
}


def rule_score(frame: pd.DataFrame) -> pd.Series:
    """总分 0–7 分；分数相同的人按原顺序排，不做随机。"""
    score = pd.Series(0.0, index=frame.index, name="rule_score")
    score += WEIGHTS["上次成功"] * (frame[S.POUTCOME] == "成功").astype("float64")
    score += WEIGHTS["被联系过"] * (frame[S.CONTACTED].astype("int8") == 1).astype("float64")
    score += WEIGHTS["手机"] * (frame[S.CONTACT] == "手机").astype("float64")
    score += WEIGHTS["高等教育"] * (frame[S.EDU] == "高等教育").astype("float64")
    score += WEIGHTS["联系次数少"] * (frame[S.CAMPAIGN] <= 2).astype("float64")
    return score
