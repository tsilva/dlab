from __future__ import annotations

from typing import Any


def build_litlogger(**kwargs: Any):
    from pytorch_lightning.loggers import LitLogger

    class QuietLitLogger(LitLogger):
        def log_graph(self, *args: Any, **kwargs: Any) -> None:
            return None

    return QuietLitLogger(**kwargs)

