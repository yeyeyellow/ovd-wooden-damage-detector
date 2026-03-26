#!/usr/bin/env python3
"""OVD Pseudo-Label Generator CLI

Command-line tool for generating pseudo-labels using Open-Vocabulary Detection.

Usage:
    python scripts/ovd_cli.py generate --images data/unlabeled/images --output data/pseudo_labels/labels
    python scripts/ovd_cli.py generate --images data/unlabeled/images --output data/pseudo_labels/labels --conf 0.3 --strategy detailed
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.ovd import (
    YOLOEModel,
    OVDConfig,
    PromptBuilder,
    PseudoLabelGenerator,
    LabelGenerationConfig,
    get_strategy,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate pseudo-labels using OVD models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate labels with default settings
  python scripts/ovd_cli.py generate --images data/unlabeled/images --output data/labels

  # Use detailed prompts and higher confidence
  python scripts/ovd_cli.py generate --images data/unlabeled/images --output data/labels --conf 0.3 --strategy detailed

  # Use custom model
  python scripts/ovd_cli.py generate --images data/unlabeled/images --output data/labels --model yolov8x-worldv2.pt
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate pseudo-labels")
    gen_parser.add_argument(
        "--images", "-i",
        required=True,
        help="Directory containing input images"
    )
    gen_parser.add_argument(
        "--output", "-o",
        required=True,
        help="Directory to save YOLO label files"
    )
    gen_parser.add_argument(
        "--model", "-m",
        default="yoloe-26x-seg.pt",
        help="OVD model name (default: yoloe-26x-seg.pt)"
    )
    gen_parser.add_argument(
        "--conf", "-c",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)"
    )
    gen_parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS (default: 0.45)"
    )
    gen_parser.add_argument(
        "--device",
        default="0",
        help="Device to run on (default: 0 for GPU)"
    )
    gen_parser.add_argument(
        "--strategy", "-s",
        choices=["basic", "detailed", "context", "expert"],
        default="expert",
        help="Prompt strategy (default: expert)"
    )
    gen_parser.add_argument(
        "--custom-prompts",
        nargs="+",
        help="Custom text prompts (overrides strategy)"
    )
    gen_parser.add_argument(
        "--report",
        help="Path to save generation report (JSON)"
    )

    return parser.parse_args()


def cmd_generate(args):
    """Generate pseudo-labels."""
    print("=" * 60)
    print("OVD Pseudo-Label Generator")
    print("=" * 60)

    # Setup OVD model
    print(f"\n[1/4] Loading OVD model: {args.model}")
    ovd_config = OVDConfig(
        model_name=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device
    )
    model = YOLOEModel(ovd_config)

    # Setup prompts
    print(f"[2/4] Setting up prompts (strategy: {args.strategy})")
    if args.custom_prompts:
        prompts = args.custom_prompts
        print(f"  Using {len(prompts)} custom prompts")
    else:
        strategy = get_strategy(args.strategy)
        prompts = strategy.generate()
        print(f"  Generated {len(prompts)} prompts:")
        for p in prompts:
            print(f"    - {p}")

    model.set_classes(prompts)

    # Setup generator
    print(f"[3/4] Initializing label generator")
    label_config = LabelGenerationConfig(
        images_dir=args.images,
        output_dir=args.output,
        conf_threshold=args.conf
    )
    generator = PseudoLabelGenerator(model, label_config)

    # Generate labels
    print(f"[4/4] Generating labels...")
    print(f"  Input:  {args.images}")
    print(f"  Output: {args.output}")
    print()

    stats = generator.generate_batch()

    # Print results
    print("\n" + "=" * 60)
    print("Generation Complete!")
    print("=" * 60)
    print(f"  Total images:      {stats['total']}")
    print(f"  Successful:        {stats['successful']}")
    print(f"  Failed:            {stats['failed']}")
    print(f"  Total detections:  {stats['total_detections']}")
    print()

    # Save report if requested
    if args.report:
        report_path = Path(args.report)
        with open(report_path, "w") as f:
            json.dump(stats, f, indent=2, default=str)
        print(f"Report saved to: {report_path}")

    return 0 if stats["failed"] == 0 else 1


def main():
    """Main entry point."""
    args = parse_args()

    if args.command == "generate":
        return cmd_generate(args)
    else:
        print("Error: No command specified.")
        print("Use 'python scripts/ovd_cli.py --help' for usage information.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
