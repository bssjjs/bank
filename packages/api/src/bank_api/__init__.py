"""bank_api：只读模型包的推理服务。

只依赖 bank（core），**不依赖** bank_training：线上不导入训练代码。
"""

from __future__ import annotations
