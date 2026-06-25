"""MoE NER 训练的命令行入口。"""

from __future__ import annotations

import argparse

from model_training.training.config import load_pipeline_config
from model_training.training.pipeline import TrainingPipeline


def parse_args() -> argparse.Namespace:
    """解析训练 workflow 的命令行参数。"""

    parser = argparse.ArgumentParser(description="Train MoE NER through the reusable pipeline.")
    parser.add_argument("--config", required=True, help="Path to pipeline YAML config.")
    parser.add_argument("--resume-from", default=None, help="Optional checkpoint to resume from.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory override.")
    return parser.parse_args()


def main() -> None:
    """加载配置，应用 workflow 级覆盖项，并启动训练。"""

    args = parse_args()
    config = load_pipeline_config(args.config)
    if args.resume_from:
        config.checkpoint.resume_from = args.resume_from
    if args.output_dir:
        config.experiment.output_dir = args.output_dir
        config.checkpoint.save_dir = args.output_dir
    TrainingPipeline(config).run()


if __name__ == "__main__":
    main()
