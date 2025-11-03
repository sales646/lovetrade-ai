#!/usr/bin/env python3
"""
Model Versioning and Management System
Tracks, compares, and deploys trained models
"""

import os
import json
import shutil
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for a trained model"""
    version: str
    created_at: str
    model_type: str  # 'bc', 'ppo', 'ensemble'
    hyperparameters: Dict
    performance_metrics: Dict
    training_data: Dict  # symbols, timeframe, date range
    checksum: str
    file_path: str
    parent_version: Optional[str] = None
    notes: str = ""
    deployed: bool = False


class ModelRegistry:
    """
    Central registry for managing model versions
    """
    
    def __init__(self, registry_dir: str = "models/registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        self.registry_file = self.registry_dir / "registry.json"
        self.models: Dict[str, ModelMetadata] = {}
        
        # Load existing registry
        self._load_registry()
    
    def _load_registry(self):
        """Load registry from disk"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                data = json.load(f)
                self.models = {
                    version: ModelMetadata(**metadata)
                    for version, metadata in data.items()
                }
            logger.info(f"Loaded {len(self.models)} models from registry")
    
    def _save_registry(self):
        """Save registry to disk"""
        data = {
            version: asdict(metadata)
            for version, metadata in self.models.items()
        }
        with open(self.registry_file, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info("Registry saved")
    
    def _generate_version(self, model_type: str) -> str:
        """Generate version string: {model_type}-{timestamp}"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{model_type}-{timestamp}"
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate MD5 checksum of model file"""
        md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                md5.update(chunk)
        return md5.hexdigest()
    
    def register_model(
        self,
        model_path: str,
        model_type: str,
        hyperparameters: Dict,
        performance_metrics: Dict,
        training_data: Dict,
        parent_version: Optional[str] = None,
        notes: str = ""
    ) -> str:
        """
        Register a new model in the registry
        
        Args:
            model_path: Path to model checkpoint file
            model_type: Type of model ('bc', 'ppo', 'ensemble')
            hyperparameters: Dict of hyperparameters used
            performance_metrics: Dict of performance metrics (sharpe, win_rate, etc.)
            training_data: Dict describing training data (symbols, timeframe, dates)
            parent_version: Version of parent model (for fine-tuned models)
            notes: Optional notes about this model
        
        Returns:
            version: Version string for registered model
        """
        # Generate version
        version = self._generate_version(model_type)
        
        # Copy model to registry
        registry_model_path = self.registry_dir / f"{version}.pt"
        shutil.copy(model_path, registry_model_path)
        
        # Calculate checksum
        checksum = self._calculate_checksum(str(registry_model_path))
        
        # Create metadata
        metadata = ModelMetadata(
            version=version,
            created_at=datetime.now().isoformat(),
            model_type=model_type,
            hyperparameters=hyperparameters,
            performance_metrics=performance_metrics,
            training_data=training_data,
            checksum=checksum,
            file_path=str(registry_model_path),
            parent_version=parent_version,
            notes=notes,
            deployed=False
        )
        
        # Add to registry
        self.models[version] = metadata
        self._save_registry()
        
        logger.info(f"✓ Registered model: {version}")
        logger.info(f"  Performance: {performance_metrics}")
        
        return version
    
    def get_model(self, version: str) -> Optional[ModelMetadata]:
        """Get model metadata by version"""
        return self.models.get(version)
    
    def list_models(
        self,
        model_type: Optional[str] = None,
        deployed_only: bool = False,
        sort_by: str = 'created_at'
    ) -> List[ModelMetadata]:
        """
        List models with optional filtering
        
        Args:
            model_type: Filter by model type
            deployed_only: Only show deployed models
            sort_by: Sort by field ('created_at', 'performance', etc.)
        """
        models = list(self.models.values())
        
        # Filter by type
        if model_type:
            models = [m for m in models if m.model_type == model_type]
        
        # Filter deployed
        if deployed_only:
            models = [m for m in models if m.deployed]
        
        # Sort
        if sort_by == 'created_at':
            models.sort(key=lambda m: m.created_at, reverse=True)
        elif sort_by == 'sharpe':
            models.sort(
                key=lambda m: m.performance_metrics.get('sharpe_ratio', 0),
                reverse=True
            )
        elif sort_by == 'win_rate':
            models.sort(
                key=lambda m: m.performance_metrics.get('win_rate', 0),
                reverse=True
            )
        
        return models
    
    def get_best_model(
        self,
        metric: str = 'sharpe_ratio',
        model_type: Optional[str] = None
    ) -> Optional[ModelMetadata]:
        """Get best model by a specific metric"""
        models = self.list_models(model_type=model_type)
        
        if not models:
            return None
        
        return max(
            models,
            key=lambda m: m.performance_metrics.get(metric, float('-inf'))
        )
    
    def compare_models(
        self,
        version1: str,
        version2: str
    ) -> Dict:
        """Compare two models"""
        model1 = self.get_model(version1)
        model2 = self.get_model(version2)
        
        if not model1 or not model2:
            raise ValueError("One or both models not found")
        
        comparison = {
            'version1': version1,
            'version2': version2,
            'metrics_comparison': {}
        }
        
        # Compare metrics
        for metric in model1.performance_metrics.keys():
            val1 = model1.performance_metrics.get(metric, 0)
            val2 = model2.performance_metrics.get(metric, 0)
            
            comparison['metrics_comparison'][metric] = {
                'model1': val1,
                'model2': val2,
                'diff': val2 - val1,
                'better': 'model2' if val2 > val1 else 'model1'
            }
        
        return comparison
    
    def deploy_model(self, version: str, deployment_path: str):
        """
        Deploy a model to production
        
        Args:
            version: Model version to deploy
            deployment_path: Path where to deploy the model
        """
        model = self.get_model(version)
        if not model:
            raise ValueError(f"Model {version} not found")
        
        # Copy to deployment location
        shutil.copy(model.file_path, deployment_path)
        
        # Mark as deployed
        model.deployed = True
        self._save_registry()
        
        logger.info(f"✓ Deployed model {version} to {deployment_path}")
    
    def delete_model(self, version: str):
        """Delete a model from registry"""
        model = self.get_model(version)
        if not model:
            raise ValueError(f"Model {version} not found")
        
        # Delete file
        if Path(model.file_path).exists():
            Path(model.file_path).unlink()
        
        # Remove from registry
        del self.models[version]
        self._save_registry()
        
        logger.info(f"✗ Deleted model {version}")
    
    def export_model_report(self, version: str, output_path: str):
        """Export detailed model report as JSON"""
        model = self.get_model(version)
        if not model:
            raise ValueError(f"Model {version} not found")
        
        report = asdict(model)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Exported model report to {output_path}")
    
    def get_model_lineage(self, version: str) -> List[str]:
        """Get the lineage (parent chain) of a model"""
        lineage = [version]
        current = self.get_model(version)
        
        while current and current.parent_version:
            lineage.append(current.parent_version)
            current = self.get_model(current.parent_version)
        
        return lineage


def create_model_card(metadata: ModelMetadata) -> str:
    """Generate a human-readable model card"""
    card = f"""
# Model Card: {metadata.version}

## Basic Information
- **Model Type**: {metadata.model_type}
- **Created**: {metadata.created_at}
- **Deployed**: {'Yes' if metadata.deployed else 'No'}
- **Parent Model**: {metadata.parent_version or 'None'}

## Performance Metrics
"""
    for metric, value in metadata.performance_metrics.items():
        card += f"- **{metric}**: {value}\n"
    
    card += f"""
## Training Data
- **Symbols**: {metadata.training_data.get('symbols', 'N/A')}
- **Timeframe**: {metadata.training_data.get('timeframe', 'N/A')}
- **Date Range**: {metadata.training_data.get('date_range', 'N/A')}

## Hyperparameters
"""
    for param, value in metadata.hyperparameters.items():
        card += f"- **{param}**: {value}\n"
    
    if metadata.notes:
        card += f"""
## Notes
{metadata.notes}
"""
    
    card += f"""
## Technical Details
- **File Path**: {metadata.file_path}
- **Checksum (MD5)**: {metadata.checksum}
"""
    
    return card
