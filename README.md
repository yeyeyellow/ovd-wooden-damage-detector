# OVD Wooden Damage Detector

Open-Vocabulary Detection (OVD) based pseudo-label generator for wooden building damage detection.

## Features

- **Zero-Shot Detection**: Use YOLOE (YOLO-World) for open-vocabulary object detection
- **Pseudo-Label Generation**: Automatically generate YOLO format labels from text prompts
- **Multiple Prompt Strategies**: Basic, Detailed, and Context-aware strategies
- **CLI Tool**: Easy-to-use command-line interface
- **94% Test Coverage**: Fully tested with pytest

## Installation

```bash
# Clone repository
git clone https://github.com/your-username/ovd-wooden-damage-detector.git
cd ovd-wooden-damage-detector

# Install dependencies
pip install ultralytics pillow pytest
```

## Quick Start

### Command Line

```bash
# Generate pseudo-labels with default settings
python ovd_cli.py generate \
    --images data/unlabeled/images \
    --output data/pseudo_labels/labels

# Use detailed prompts
python ovd_cli.py generate \
    --images data/unlabeled/images \
    --output data/pseudo_labels/labels \
    --strategy detailed

# Use custom prompts
python ovd_cli.py generate \
    --images data/unlabeled/images \
    --output data/pseudo_labels/labels \
    --custom-prompts "wood crack" "wood decay" "insect holes"
```

### Python API

```python
from src.ovd import YOLOEModel, PseudoLabelGenerator, LabelGenerationConfig, get_strategy

# Initialize model
model = YOLOEModel()
prompts = get_strategy("detailed").generate()
model.set_classes(prompts)

# Generate labels
config = LabelGenerationConfig(
    images_dir="data/unlabeled/images",
    output_dir="data/pseudo_labels/labels"
)
generator = PseudoLabelGenerator(model, config)
stats = generator.generate_batch()

print(f"Generated labels for {stats['successful']}/{stats['total']} images")
```

## Damage Classes

| Class ID | Name | Description |
|----------|------|-------------|
| 0 | crack | Wood cracks or fractures |
| 1 | decay | Decayed or rotten wood |
| 2 | insect | Insect holes or pest damage |
| 3 | mechanical | Mechanical damage or tool marks |
| 4 | mildew | Mold or fungal growth |
| 5 | knot | Wood knots or natural nodes |

## Prompt Strategies

### Basic
Simple class names: `["crack", "decay", "insect", ...]`

### Detailed
Descriptive phrases: `["wood crack or fracture on timber surface", ...]`

### Context
Building context: `["traditional wooden building damage: crack on timber beam", ...]`

## Output Format

YOLO format label files:
```
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are normalized to [0, 1].

## Requirements

- Python 3.8+
- ultralytics
- Pillow
- pytest (for testing)

## License

MIT License

## Acknowledgments

- [YOLO-World](https://github.com/AILab-CVC/YOLO-World) - Open-vocabulary detection model
- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLO implementation
