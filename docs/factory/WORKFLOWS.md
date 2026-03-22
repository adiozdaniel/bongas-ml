# ⚙️ General Workflows: The Production Line

[🏠 Hub](../README.md) | [🏗️ Factory](./WORKFLOWS.md)

---

This document outlines the internal **Vendor-Only** processes for preparing a model release.

---

## 🛠️ Internal Tools

### 1. Exporters (`factory/exporters/`)

Used to convert massive PyTorch foundation models into optimized, high-integrity .safetensors base models.

* **Logic:** `export_base_model.py`
* **Guide:** [📦 Exporters Detailed Guide](./EXPORTERS.md)

### 2. Builders (`factory/builders/`)

The **IP-Protection** machinery that compiles our proprietary Python training loops into C-extensions.

* **Logic:** `compile_trainer.py` (Cython)
* **Guide:** [🛡️ Builders Detailed Guide](./BUILDERS.md)

---

## 🚀 The Build & Ship Pipeline

To prepare a full release for the client, follow this sequence:

1. **Refine Student Heads:** Modify training logic in `factory/trainer/`. See [🧠 Trainer Guide](./TRAINER.md).
2. **Export Base:** Run exporters to get latest optimized "Senses".
3. **Compile Blackbox:** Run builders to obfuscate the training loop.
4. **Package Artifacts:** Move all results into `bongas-ai/release/`.

---

## ⚠️ Important Note

**NEVER** ship any file from the `research/` or `factory/` folders to the client. Only the compiled outputs in `dist/` are safe for on-premise deployment.

---
[🏠 Hub](../README.md) | [🔝 Top](#️-general-workflows-the-production-line)
