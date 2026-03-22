# 🧬 Sovereign Discovery: The Machine Learning Engine

> **The Vendor Safe Haven: Secure, Offline Development & Obfuscation Layer.**

Welcome to **BONGAS-ML**. This repository is the high-security environment for pre-training massive foundation models and defining the intelligence that powers the Bongas-AI ecosystem.

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
            Obfuscator["Binary Obfuscation<br/>(IP Protection)"]
            VendorImage["Docker Image: bongas-ai-sovereign:latest<br/>(Single Binary Engine)"]
        end

        %% Force Vertical Stack in LR Graph
        CentralRegistry ~~~ VendorEnv
    end

    %% Client Sovereign Environment
    subgraph ClientVPC ["Client Sovereign VPC (On-Premise)"]
        RawContent["Raw Videos / Content"]
        ClickHouse["ClickHouse<br/>(Feedback & Ledgers)"]
        BongasAIBin["Bongas-AI Binary<br/>(Security & Intelligence)"]
        MLContainer["Asynchronous ML Sidecar<br/>(Docker Container)"]
        
        subgraph SovereignEngine ["Sovereign Training Engine"]
            NativeTrainer["Native Rust Trainer<br/>(Integrated in binary)"]
            ExecSenses["weights.safetensors<br/>(Frozen Inference)"]
            
            VisionTuned["vision_head.safetensors<br/>(Fine-Tuned)"]
            SLMTuned["slm_head.safetensors<br/>(Fine-Tuned)"]
            FlowTuned["flow_head.safetensors<br/>(Fine-Tuned)"]
            RankingTuned["ranking_head.safetensors<br/>(Fine-Tuned)"]
        end
    end

    %% --- CONNECTIONS ---
    %% Vendor Factory
    GoldenData --> PreTraining
    PreTraining --> Obfuscator
    Obfuscator --> VendorImage
    PreTraining -.-> VendorImage
    
    %% Export to Server
    VendorImage --> RegistryAPI

    %% Server to VPC Delivery (Docker Pull)
    RegistryAPI --->|Docker Pull Layered| MLContainer
    MLContainer -->|Top Layer| BongasAIBin
    MLContainer -->|Base Layer| ExecSenses
    MLContainer -->|Base Layer| VisionTuned
    MLContainer -->|Base Layer| SLMTuned
    MLContainer -->|Base Layer| FlowTuned
    MLContainer -->|Base Layer| RankingTuned

    %% Sovereign Orchestration & Security
    BongasAIBin <--->|mTLS Heart Beat & Keys| RegistryAPI
    
    %% Sensing & Feedback Loop
    RawContent --> ExecSenses
    ExecSenses -->|Semantic DNA| ClickHouse
    ClickHouse -->|Ledgers| BongasAIBin
    
    %% Sovereign Training
    BongasAIBin --> VisionTuned
    BongasAIBin --> SLMTuned
    BongasAIBin --> FlowTuned
    BongasAIBin --> RankingTuned
```

To balance **Data Sovereignty** with **Intellectual Property Protection**, this engine operates on a bifurcated architecture:

* **👁️ Frozen Senses:** 1.2B+ parameter models exported as read-only Safetensors.
* **🎓 Student Heads:** Minimal layers trained locally on private ClickHouse telemetry via the native Rust Training Pillar.
* **🛡️ Integrated Security:** The compiled binary handles its own integrity checks, heartbeats, and silent updates.

## 📚 Documentation Hub

Explore the full architectural blueprints and production workflows in our central knowledge base:

👉 **[Enter the Documentation Hub](./docs/README.md)**

---
[🏠 Home](./README.md) | [📚 Hub](./docs/README.md) | [🏗️ Factory](./docs/factory/WORKFLOWS.md)

### *BONGAS-AI - Unified Content Intelligence Orchestration for the Global Digital Economy*
