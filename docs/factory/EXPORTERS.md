# 📦 Factory Exporters: Preparing Foundation Models

[🏠 Hub](../README.md) | [🏗️ Factory](./WORKFLOWS.md) | [📦 Exporters](./EXPORTERS.md)

---

## The Goal

Eliminate PyTorch as a dependency on the client's production server by converting foundation models to Safetensors.

## Standard Export Workflow

1. **Place Research Assets:** Ensure the raw model exists in `research/vision/assets/`.
2. **Define Dummy Input:** Use matching tensor shapes (e.g., `[batch, channels, frames, height, width]`).
3. **Trace and Export:**

    ```bash
    python factory/exporters/export_base_model.py
    ```

4. **Extract Artifacts:** High-integrity .safetensors weights are deposited into `../bongas-ai/release/models/frozen/`.
5. **Copy Preprocessors:** Critical `.json` configs are copied automatically.

---
[🏠 Hub](../README.md) | [🔝 Top](#-factory-exporters-preparing-foundation-models)
