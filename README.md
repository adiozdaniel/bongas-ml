# 🧬 Sovereign Discovery: The Machine Learning Engine

> **The Vendor Safe Haven: Secure, Offline Development & Obfuscation Layer.**

Welcome to **BONGAS-ML**. This repository is the high-security environment for pre-training massive foundation models and compiling the "Blackbox" intelligence that powers the Bongas-AI ecosystem.

---

## 🏛️ The Sovereign Paradigm

👉 **[View the Interactive Bongas-ML Architecture Diagram](./docs/architecture/BONGAS_ML_INTERACTIVE_DIAGRAM.html)**
> ⚠️ **GitLab/GitHub Notice:** By default, repository platforms render `.html` files as raw source code for security reasons. To view the animations and interactive elements, please **download** the `BONGAS_ML_INTERACTIVE_DIAGRAM.html` file to your computer and open it in your web browser.

```mermaid
graph LR
    subgraph VendorRegion ["<br/><br/><br/>Vendor Infrastructure & Registry"]
        direction TB
        %% Bongas-Server (Central Middleman)
        subgraph CentralRegistry ["Bongas-Server (Central Registry)"]
            RegistryAPI["Registry Core API<br/>(Distribution & Models)"]
        end

        %% Vendor Environment
        subgraph VendorEnv ["Bongas-ML Factory (Vendor Server)"]
            GoldenData["Golden Datasets<br/>(Global Content)"]
            PreTraining["Foundation Pre-Training<br/>(Heavy Compute)"]
            Obfuscator["Cython Compiler<br/>(IP Protection)"]
            VendorBundle["bongas-ml-bundle.tar.gz<br/>(All Compiled Artifacts)"]
        end

        %% Force Vertical Stack in LR Graph
        CentralRegistry ~~~ VendorEnv
    end

    %% Client Sovereign Environment
    subgraph ClientVPC ["Client Sovereign VPC (On-Premise)"]
        RawContent["Raw Videos / Content"]
        ClickHouse["ClickHouse<br/>(Feedback & Ledgers)"]
        BongasAIBin["Bongas-AI Binary<br/>(Security)"]
        ClientBundle["Downloaded Bundle<br/>(bongas-ml-bundle.tar.gz)"]
        
        subgraph SovereignEngine ["Sovereign Training Engine"]
            ExecTrainer["trainer.so<br/>(Orchestrator & Execution)"]
            ExecSenses["weights.safetensors<br/>(Frozen Inference)"]
            
            VisionTuned["vision_head.onnx<br/>(Fine-Tuned)"]
            SLMTuned["slm_head.onnx<br/>(Fine-Tuned)"]
            FlowTuned["flow_head.onnx<br/>(Fine-Tuned)"]
            RankingTuned["ranking.onnx<br/>(Fine-Tuned)"]
        end
    end

    %% --- CONNECTIONS ---
    %% Vendor Factory
    GoldenData --> PreTraining
    PreTraining --> Obfuscator
    Obfuscator --> VendorBundle
    PreTraining -.-> VendorBundle
    
    %% Export to Server
    VendorBundle --> RegistryAPI

    %% Server to VPC Delivery (Single Download)
    RegistryAPI --->|Client Downloads Bundle| ClientBundle
    ClientBundle -->|Unpacks| ExecTrainer
    ClientBundle -->|Unpacks| ExecSenses
    ClientBundle -->|Unpacks Base| VisionTuned
    ClientBundle -->|Unpacks Base| SLMTuned
    ClientBundle -->|Unpacks Base| FlowTuned
    ClientBundle -->|Unpacks Base| RankingTuned

    %% Sovereign Orchestration & Security
    ExecTrainer -->|Verifies Security| BongasAIBin
    ExecTrainer <--->|Heart Beat, Updates & Security Checks| RegistryAPI
    
    %% Sensing & Feedback Loop
    RawContent --> ExecSenses
    ExecSenses -->|Semantic DNA| ClickHouse
    ClickHouse -->|Ledgers| ExecTrainer
    
    %% Sovereign Training
    ExecTrainer --> VisionTuned
    ExecTrainer --> SLMTuned
    ExecTrainer --> FlowTuned
    ExecTrainer --> RankingTuned
```

To balance **Data Sovereignty** with **Intellectual Property Protection**, this engine operates on a bifurcated architecture:

* **👁️ Frozen Senses:** 1.2B+ parameter models exported as read-only ONNX graphs.
* **🎓 Student Heads:** Minimal layers trained locally on private ClickHouse telemetry.
* **🛡️ trainer.so:** A Cythonized, obfuscated orchestrator that keeps our math secret.

## 📚 Documentation Hub

Explore the full architectural blueprints and production workflows in our central knowledge base:

👉 **[Enter the Documentation Hub](./docs/README.md)**

---
[🏠 Home](./README.md) | [📚 Hub](./docs/README.md) | [🏗️ Factory](./docs/factory/WORKFLOWS.md)

### *BONGAS-AI - Unified Content Intelligence Orchestration for the Global Digital Economy*
