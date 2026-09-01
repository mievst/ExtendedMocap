"""Command-line entry points for the ``extended-mocap`` console script."""

from __future__ import annotations

import argparse
import sys


def _run_extract(args) -> int:
    from .extractor import MediapipeExtractor

    extractor = MediapipeExtractor()
    poses = extractor.run(args.video, output_csv_path=args.output)
    print(f"Extracted {len(poses)} frames of features")
    return 0


def _run_infer(args) -> int:
    from .inference import DEFAULT_MODEL_CONFIG, MocapInferer

    config = args.config or DEFAULT_MODEL_CONFIG
    inferer = MocapInferer(model_config=config)
    out = inferer.predict(args.features_csv)
    out.to_csv(args.output, index=False)
    print(f"Predicted {len(out)} frames -> {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="extended-mocap", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    ex = sub.add_parser("extract", help="Extract MediaPipe features from a video")
    ex.add_argument("video", help="Input video path")
    ex.add_argument("-o", "--output", default=None, help="Output feature CSV path")
    ex.set_defaults(func=_run_extract)

    inf = sub.add_parser("infer", help="Predict quaternions from a feature CSV")
    inf.add_argument("features_csv", help="Feature CSV produced by 'extract'")
    inf.add_argument("-o", "--output", default="motion.csv", help="Output quaternion CSV path")
    inf.add_argument("--config", default=None, help="Optional JSON model config")
    inf.set_defaults(func=_run_infer)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
