from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io import load_dynamic_events, load_tracking
from .features import build_shot_temporal_features
from .quality import tracking_quality_report
from .validation import shot_validation_summary, validate_shot_defender_distance


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SkillCorner basketball data utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser("inspect", help="Generate a tracking quality report")
    inspect.add_argument("tracking_path")
    inspect.add_argument("--game-id", required=True)
    inspect.add_argument("--output")

    normalize = subparsers.add_parser("normalize", help="Write canonical frame/entity CSVs")
    normalize.add_argument("tracking_path")
    normalize.add_argument("--game-id", required=True)
    normalize.add_argument("--output-dir", required=True)

    validate = subparsers.add_parser(
        "validate-shots", help="Recompute closest-defender distance at shot release"
    )
    validate.add_argument("tracking_path")
    validate.add_argument("events_path")
    validate.add_argument("--game-id", required=True)
    validate.add_argument("--output-dir", required=True)

    features = subparsers.add_parser(
        "build-shot-features", help="Build one temporal feature row per shot"
    )
    features.add_argument("tracking_path")
    features.add_argument("events_path")
    features.add_argument("--game-id", required=True)
    features.add_argument("--output", required=True)
    features.add_argument("--window-seconds", type=float, default=2.0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    frames, entities = load_tracking(args.tracking_path, args.game_id)

    if args.command == "inspect":
        report = tracking_quality_report(frames, entities)
        rendered = json.dumps(report, ensure_ascii=False, indent=2)
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return

    if args.command == "build-shot-features":
        events = load_dynamic_events(args.events_path)
        features = build_shot_temporal_features(
            events["shots"],
            entities,
            game_id=args.game_id,
            window_seconds=args.window_seconds,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(output, index=False)
        print(
            json.dumps(
                {
                    "shot_count": int(len(features)),
                    "valid_count": int((features["status"] == "ok").sum()),
                    "output": str(output),
                },
                indent=2,
            )
        )
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "validate-shots":
        events = load_dynamic_events(args.events_path)
        if "shots" not in events:
            raise ValueError("Dynamic events bundle does not contain a shots table")
        validation = validate_shot_defender_distance(events["shots"], entities)
        validation.to_csv(output_dir / "shot_distance_validation.csv", index=False)
        summary = shot_validation_summary(validation)
        (output_dir / "shot_distance_validation_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    frames.to_csv(output_dir / "frames.csv", index=False)
    entities.to_csv(output_dir / "entities.csv", index=False)
    print(f"Saved {len(frames)} frames and {len(entities)} entities to {output_dir}")


if __name__ == "__main__":
    main()
