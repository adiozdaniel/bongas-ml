# 🧠 Factory Trainer: Refining the Student Loops

[🏠 Hub](../README.md) | [🏗️ Factory](./WORKFLOWS.md) | [🧠 Trainer](./TRAINER.md)

---

## The Concept

Training lightweight "Student Heads" locally on-premise using ClickHouse interaction ledgers while maintaining Data Sovereignty.

## Trainer Architecture (`factory/trainer/`)

### 1. Student Heads (`models.py`)

* `VisionAuditorHead`: DNA -> Safety/Vibe labels.
* `TribeConductorHead`: Tribe + Content -> Interaction Probability.

### 2. The Orchestrator (`orchestrator.py`)

1. **Fetch Local Ledger:** Queries `sovereign_sight_ledger`.
2. **Optimize:** Standard PyTorch training loop.
3. **Safetensors Export:** Serializes the trained network weights for native Rust execution.

---
[🏠 Hub](../README.md) | [🔝 Top](#-factory-trainer-refining-the-student-loops)
