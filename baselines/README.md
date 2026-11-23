# baselines/：第一轮结果的只读参照

这里的文件**不是**代码，是"第一轮跑通时结果长什么样"的快照，用来在以后数字对不上时做对账。

## 里面有什么

| 文件 | 内容 |
| --- | --- |
| `SHA256SUMS` | 对 `v1/` 下每个文件的文件封条（sha256 + 相对路径，按路径排序、正斜杠、LF） |
| `v1/runs/m3-*.json` | 四个训练 run 的指标与参数（规则基线、LR 主模型、LR-A、LR-B、RF-A） |
| `v1/experiments.json` | E1–E5 五个对照实验的指标与 run_id |
| `v1/frozen.json` | 冻结账单副本（行数、正例数、内容指纹、源文件指纹） |
| `v1/bundle-<版本>.json` | 当前模型包的 bundle.json 副本（含 checksum 与训练来源） |

## 怎么用

```powershell
.venv\Scripts\python.exe scripts\freeze_baseline.py --check   # 基线没被改过才算过
```

`--check` 分三类报：缺失、内容变了、多出（不在封条里）。质量门里跑的就是这一条。

## 什么时候可以重写

只有一种情况：**你确实改变了结果，并且知道为什么**——换了主模型、改了特征集、改了窗口或切分。
这时跑：

```powershell
.venv\Scripts\python.exe scripts\freeze_baseline.py --write
```

然后在提交信息里写清"为什么基线变了"。人手里改这里的 JSON 一律算篡改：那等于把参照物
改成对自己有利的样子，后面所有对账都失去意义。

## 为什么快照里没有 model.joblib

模型文件的 sha256 已经写在 `bundle-<版本>.json` 的 `model_file` 字段里；把 10 KB 的
`bundle.json` 封进 Git 就够核对，不必再把模型二进制复制一份（RF 对照模型有 14 MB）。