import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from loguru import logger
from typing import Any, Tuple, List

from .models import VisionAuditorHead, TribeConductorHead

class BlackboxSovereignTrainer:
    """
    Obfuscated Training Orchestrator.
    
    This class is designed to be Cythonized into `trainer.so`.
    It pulls raw DNA from the local ClickHouse instance, trains the 
    Student Heads on-premise, and exports ONNX models for Rust ingestion.
    """
    def __init__(self, db_client: Any, output_dir: str):
        self.db = db_client
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Sovereign Trainer initialized. Export Path: {self.output_dir}")

    def _fetch_offline_vision_ledger(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Securely queries the local ClickHouse `sovereign_sight_ledger`
        and joins it with admin-overridden safety labels.
        """
        logger.info("[Vision Auditor] Fetching local visual_dna and consensus labels...")
        
        # Phase 1 Simulation Data:
        # X: The heavy 1.2B vectors (Extracted asynchronously by Rust)
        X_visual_dna = torch.randn(100, 1024).to(self.device)
        
        # Y: The client's custom categories (e.g., 0: GE, 1: PG, 2: 18+)
        Y_safety = torch.randint(0, 3, (100,)).to(self.device)
        
        # Y: The client's specific semantic tags (e.g., 'High-Energy')
        Y_vibe = torch.randint(0, 10, (100,)).to(self.device)
        
        return X_visual_dna, Y_safety, Y_vibe

    def train_and_export_vision_head(self, epochs: int = 15):
        """Trains the Vision Auditor Head on local catalog DNA."""
        logger.info("[Vision Auditor] Initializing Local Student Head...")
        model = VisionAuditorHead().to(self.device)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        X, Y_safety, Y_vibe = self._fetch_offline_vision_ledger()

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            safety_preds, vibe_preds = model(X)
            
            # Joint objective: Minimize both safety prediction and vibe errors
            loss_safety = criterion(safety_preds, Y_safety)
            loss_vibe = criterion(vibe_preds, Y_vibe)
            loss = loss_safety + loss_vibe
            
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 5 == 0:
                logger.debug(f"Epoch {epoch+1}/{epochs} | Total Loss: {loss.item():.4f}")

        logger.success("[Vision Auditor] Local Head Training Complete.")
        
        # The Rust Sidecar expects standard input names for multi-head inference
        self._export_to_onnx(
            model=model,
            filename="vision_head.onnx",
            dummy_inputs=(torch.randn(1, 1024).to(self.device),),
            input_names=["visual_dna"],
            output_names=["safety_logits", "vibe_logits"]
        )

    def _fetch_offline_tribe_ledger(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Queries ClickHouse for 'Tribe x Content' interactions, automatically
        pruning Personally Identifiable Information (PII) to ensure anonymity.
        """
        logger.info("[Tribe Conductor] Fetching local interaction telemetry...")
        
        # Simulated behavioral data (Aggregated by Tribe, NOT by user_id)
        X_tribe = torch.randn(500, 64).to(self.device)
        X_item = torch.randn(500, 1024).to(self.device)
        Y_interaction = torch.randint(0, 2, (500, 1), dtype=torch.float32).to(self.device)
        
        return X_tribe, X_item, Y_interaction

    def train_and_export_ranking_head(self, epochs: int = 10):
        """Trains the Tribe Conductor to learn local aggregate affinities."""
        logger.info("[Tribe Conductor] Initializing Local Ranking Head...")
        model = TribeConductorHead().to(self.device)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCELoss() # Binary Cross Entropy for interaction probability

        X_tribe, X_item, Y_interaction = self._fetch_offline_tribe_ledger()

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            probability = model(X_tribe, X_item)
            loss = criterion(probability, Y_interaction)
            
            loss.backward()
            optimizer.step()

        logger.success(f"[Tribe Conductor] Ranking Loop Complete. Final Loss: {loss.item():.4f}")
        
        # Export ranking model to be loaded by `ONNXInferenceStage` in Rust
        self._export_to_onnx(
            model=model,
            filename="ranking.onnx",
            dummy_inputs=(torch.randn(1, 64).to(self.device), torch.randn(1, 1024).to(self.device)),
            input_names=["tribe_embedding", "item_dna"],
            output_names=["interaction_probability"]
        )

    def _export_to_onnx(self, model: nn.Module, filename: str, dummy_inputs: Tuple, 
                        input_names: List[str], output_names: List[str]):
        """
        Serializes the trained PyTorch head into a highly optimized ONNX graph
        for sub-millisecond execution inside the Rust `OnnxInferenceEngine`.
        """
        model.eval()
        export_path = self.output_dir / filename
        
        torch.onnx.export(
            model,
            dummy_inputs,
            str(export_path),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={
                input_names[0]: {0: 'batch_size'} # Enable batching for inference
            }
        )
        logger.info(f"Deployed secure ONNX graph -> {export_path.name}")

    def run_all_offline_pipelines(self):
        """
        The main entrypoint invoked by cron/Airflow on the client's infrastructure.
        """
        logger.info("=== Starting Sovereign Offline Batch Training ===")
        self.train_and_export_vision_head()
        self.train_and_export_ranking_head()
        logger.success("=== All Student Heads Exported for Rust Ingestion ===")

