# BONGAS-ML

Machine Learning package for the BONGAS-AI recommendation system.

## Overview

BONGAS-ML is the **Vendor Safe Haven** for the Bongas-AI recommendation system. It serves as the secure, offline environment where massive foundation models are pre-trained, and where the obfuscated "Blackbox" components for on-premise deployments are built.

To balance **Data Sovereignty** with **Intellectual Property Protection**, BONGAS-ML facilitates the **Sovereign Discovery** architecture:

- **Frozen Base Models**: Massive pre-trained foundation models (e.g., `sight-core` for visual DNA extraction, `slm-base` for natural language reasoning) are prepared here and shipped to clients as read-only `.safetensors`.
- **Blackbox Local Trainer**: BONGAS-ML compiles a lightweight, Cythonized `trainer.so` worker. This worker is deployed to the client's on-premise infrastructure to securely train the final "Student Heads" without exfiltrating their private interaction data.
- **Local Student Heads**: The `trainer.so` uses the local ClickHouse telemetry and the Frozen Base Models to output three specific ONNX models for the Rust inference engine:
  - `vision_head.onnx`: Maps raw visual DNA to specific safety ceilings (18+, Kids) and semantic vibes.
  - `slm_head.onnx`: Generates natural language summaries and taxonomy tags.
  - `ranking.onnx`: Learns aggregate Tribe-level content affinities.

## Installation

```bash
pip install bongas-ml
```

Or install from source:

```bash
git clone https://github.com/Bongas-Squad/bongas-ml.git
cd bongas-ml
pip install -e .
```

## Quick Start

### Training a Model

```python
from bongas_ml import TwoTowerModel, Trainer, TrainingDataset

# Create model
model = TwoTowerModel(
    user_feature_dim=128,
    item_feature_dim=64,
    embedding_dim=32
)

# Create trainer
trainer = Trainer(
    model=model,
    learning_rate=0.001,
    batch_size=256,
    epochs=100
)

# Train model
trainer.train(train_data, val_data)

# Save model
model.save("model.pth")
```

### Exporting to ONNX

```python
from bongas_ml import ONNXExporter

exporter = ONNXExporter()
result = exporter.export(
    model_path="model.pth",
    output_path="model.onnx",
    optimize=True,
    quantize=True
)
```

### Model Validation

```python
from bongas_ml import AccuracyValidator

validator = AccuracyValidator()
passed, metrics = validator.validate(model, test_data)

print(f"Validation passed: {passed}")
print(f"Metrics: {metrics}")
```

### Model Registry Code

```python
from bongas_ml import ModelRegistryClient

client = ModelRegistryClient(
    registry_url="https://registry.bongas.ai",
    api_key="your-api-key"
)

# Upload model
result = client.upload_model(
    customer_id="customer123",
    model_path="model.onnx",
    metadata={"description": "Production model"}
)

# Download model
client.download_model(
    customer_id="customer123",
    output_path="downloaded_model.onnx"
)
```

## CLI Usage

### Train a Model

```bash
python -m bongas_ml train \
    --data data.csv \
    --model-type two_tower \
    --epochs 100 \
    --output model.pth
```

### Export Model

```bash
python -m bongas_ml export \
    --model model.pth \
    --output model.onnx \
    --optimize \
    --quantize
```

### Validate Model

```bash
python -m bongas_ml validate \
    --model model.onnx \
    --test-data test.csv
```

### Registry Operations

```bash
# Upload model
python -m bongas_ml registry upload \
    --model model.onnx \
    --customer customer123 \
    --registry-url https://registry.bongas.ai

# Download model
python -m bongas_ml registry download \
    --customer customer123 \
    --output model.onnx \
    --registry-url https://registry.bongas.ai

# List models
python -m bongas_ml registry list \
    --customer customer123 \
    --registry-url https://registry.bongas.ai
```

## Architecture

```txt
bongas-ml/
├── models/           # Model definitions (TwoTower, NCF, etc.)
├── training/         # Training framework and utilities
├── features/         # Feature engineering and preprocessing
├── export/           # ONNX export and optimization
├── registry/         # Model registry client
├── validation/       # Model validation and quality assurance
├── utils/           # Utility functions and logging
└── __main__.py      # CLI entry point
```

## Supported Models

- **TwoTowerModel**: Dual encoder architecture for user-item recommendations
- **NCFModel**: Neural Collaborative Filtering
- **WideAndDeepModel**: Wide & Deep learning
- **AutoIntModel**: Attentional InteRaction network
- **DINModel**: Deep Interest Network
- **BERT4RecModel**: BERT for sequential recommendations

## Features

### Model Training

- Configurable training loops with callbacks
- Multi-GPU support
- Early stopping and model checkpointing
- Custom loss functions and metrics

### Feature Engineering

- Automatic feature extraction
- Feature transformation and normalization
- Embedding layers for categorical features
- Temporal feature handling

### Model Export

- ONNX export with optimization
- Quantization for model size reduction
- Graph optimization for inference speed
- Cross-platform compatibility

### Model Registry

- Model versioning and staging
- Customer-specific model management
- Metadata tracking and lineage
- Integration with bongas-server

### Validation

- Accuracy validation gates
- Baseline comparison testing
- Shadow testing for production validation
- Comprehensive metrics and reporting

## Configuration

Create a configuration file `config.yaml`:

```yaml
model:
  type: two_tower
  user_feature_dim: 128
  item_feature_dim: 64
  embedding_dim: 32

training:
  learning_rate: 0.001
  batch_size: 256
  epochs: 100
  validation_split: 0.2

export:
  optimize: true
  quantize: true
  target_device: cpu

registry:
  url: https://registry.bongas.ai
  api_key: your-api-key
```

## Development

### Setup Development Environment

```bash
git clone https://github.com/Bongas-Squad/bongas-ml.git
cd bongas-ml
pip install -e ".[dev,test]"
pre-commit install
```

### Running Tests

```bash
pytest
pytest --cov=bongas_ml
```

### Code Formatting

```bash
black .
flake8 .
mypy .
```

### Building Documentation

```bash
pip install -e ".[docs]"
cd docs
make html
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for your changes
5. Run the test suite
6. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Support

- Documentation: [https://bongas-ai.readthedocs.io/](https://bongas-ai.readthedocs.io/)
- Issues: [GitHub Issues](https://github.com/Bongas-Squad/bongas-ml/issues)

## Related Projects

- [bongas-ai](https://github.com/Bongas-Squad/bongas-ai): Main BONGAS-AI repository
- [bongas-server](https://github.com/Bongas-Squad/bongas-server): Model serving and management server
