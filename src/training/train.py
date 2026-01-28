#training script

# src/training/train.py
"""
Two-stage training pipeline for Azure ML Compute
Stage 1: Train heads only (backbone frozen)
Stage 2: Fine-tune backbone + heads
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import mlflow
import time
from tqdm import tqdm
import numpy as np

from model import create_model
from dataset import create_dataloaders


class MultiTaskTrainer:
    """
    Trainer for multi-task fruit freshness classification
    """
    
    def __init__(self, model, train_loader, val_loader, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss functions
        self.fruit_criterion = nn.CrossEntropyLoss()
        self.freshness_criterion = nn.CrossEntropyLoss()
        self.confidence_criterion = nn.MSELoss()
        
        # Loss weights (tunable hyperparameters)
        self.loss_weights = {
            'fruit': 0.3,
            'freshness': 0.5,
            'confidence': 0.2
        }
        
        # Metrics tracking
        self.best_val_acc = 0.0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_fruit_acc': [],
            'val_freshness_acc': [],
            'val_combined_acc': []
        }
    
    def train_stage1(self, epochs=20, lr=1e-3):
        """
        Stage 1: Train classification heads only (backbone frozen)
        Faster convergence, leverages pretrained features
        """
        print("\n" + "="*60)
        print("STAGE 1: Training classification heads (backbone frozen)")
        print("="*60)
        
        # Optimizer: only heads
        optimizer = optim.AdamW([
            {'params': self.model.fruit_classifier.parameters()},
            {'params': self.model.freshness_classifier.parameters()},
            {'params': self.model.confidence_regressor.parameters()}
        ], lr=lr, weight_decay=1e-4)
        
        # Scheduler
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        
        # Train
        for epoch in range(epochs):
            train_loss = self._train_epoch(optimizer)
            val_metrics = self._validate()
            scheduler.step()
            
            # Logging
            self._log_metrics(epoch, train_loss, val_metrics, stage=1)
            
            # Save best model
            if val_metrics['combined_acc'] > self.best_val_acc:
                self.best_val_acc = val_metrics['combined_acc']
                self._save_checkpoint('checkpoint_stage1_best.pth')
                print(f"  ✓ New best model saved (acc: {self.best_val_acc:.4f})")
        
        print(f"\nStage 1 complete. Best validation accuracy: {self.best_val_acc:.4f}")
        return self.history
    
    def train_stage2(self, epochs=10, lr=1e-4):
        """
        Stage 2: Fine-tune entire model (backbone unfrozen)
        Lower learning rate to avoid catastrophic forgetting
        """
        print("\n" + "="*60)
        print("STAGE 2: Fine-tuning entire model (backbone unfrozen)")
        print("="*60)
        
        # Unfreeze backbone (last few layers)
        self.model.unfreeze_backbone(unfreeze_from_layer=6)
        
        # Optimizer: all parameters with differential learning rates
        optimizer = optim.AdamW([
            {'params': self.model.backbone.parameters(), 'lr': lr * 0.1},  # Lower LR for backbone
            {'params': self.model.fruit_classifier.parameters(), 'lr': lr},
            {'params': self.model.freshness_classifier.parameters(), 'lr': lr},
            {'params': self.model.confidence_regressor.parameters(), 'lr': lr}
        ], weight_decay=1e-4)
        
        # Scheduler with plateau detection
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
        
        # Early stopping
        patience_counter = 0
        patience_limit = 10
        
        # Train
        for epoch in range(epochs):
            train_loss = self._train_epoch(optimizer)
            val_metrics = self._validate()
            scheduler.step(val_metrics['combined_acc'])
            
            # Logging
            self._log_metrics(epoch, train_loss, val_metrics, stage=2)
            
            # Save best model
            if val_metrics['combined_acc'] > self.best_val_acc:
                self.best_val_acc = val_metrics['combined_acc']
                self._save_checkpoint('checkpoint_stage2_best.pth')
                print(f"  ✓ New best model saved (acc: {self.best_val_acc:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience_limit:
                print(f"\nEarly stopping triggered (no improvement for {patience_limit} epochs)")
                break
        
        print(f"\nStage 2 complete. Best validation accuracy: {self.best_val_acc:.4f}")
        return self.history
    
    def _train_epoch(self, optimizer):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            # Unpack tuple from dataloader: (image, fruit_label, freshness_label)
            images, fruit_labels, freshness_labels = batch
            images = images.to(self.device)
            fruit_labels = fruit_labels.to(self.device)
            freshness_labels = freshness_labels.to(self.device)
            
            # Create confidence targets (inverse of freshness: fresh=0 -> conf=1)
            confidence_targets = (1 - freshness_labels).float().unsqueeze(1).to(self.device)
            
            # Forward pass
            fruit_logits, freshness_logits, confidence_pred = self.model(images)
            
            # Multi-task loss
            loss_fruit = self.fruit_criterion(fruit_logits, fruit_labels)
            loss_freshness = self.freshness_criterion(freshness_logits, freshness_labels)
            loss_confidence = self.confidence_criterion(confidence_pred, confidence_targets)
            
            # Weighted combination
            loss = (
                self.loss_weights['fruit'] * loss_fruit +
                self.loss_weights['freshness'] * loss_freshness +
                self.loss_weights['confidence'] * loss_confidence
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(self.train_loader)
    
    def _validate(self):
        """Validate model"""
        self.model.eval()
        
        val_loss = 0.0
        fruit_correct = 0
        freshness_correct = 0
        combined_correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Unpack tuple from dataloader: (image, fruit_label, freshness_label)
                images, fruit_labels, freshness_labels = batch
                images = images.to(self.device)
                fruit_labels = fruit_labels.to(self.device)
                freshness_labels = freshness_labels.to(self.device)
                
                # Create confidence targets
                confidence_targets = (1 - freshness_labels).float().unsqueeze(1).to(self.device)
                
                # Forward pass
                fruit_logits, freshness_logits, confidence_pred = self.model(images)
                
                # Loss
                loss_fruit = self.fruit_criterion(fruit_logits, fruit_labels)
                loss_freshness = self.freshness_criterion(freshness_logits, freshness_labels)
                loss_confidence = self.confidence_criterion(confidence_pred, confidence_targets)
                
                loss = (
                    self.loss_weights['fruit'] * loss_fruit +
                    self.loss_weights['freshness'] * loss_freshness +
                    self.loss_weights['confidence'] * loss_confidence
                )
                
                val_loss += loss.item()
                
                # Accuracy
                fruit_pred = fruit_logits.argmax(dim=1)
                freshness_pred = freshness_logits.argmax(dim=1)
                
                fruit_correct += (fruit_pred == fruit_labels).sum().item()
                freshness_correct += (freshness_pred == freshness_labels).sum().item()
                
                # Combined: both predictions must be correct
                both_correct = (fruit_pred == fruit_labels) & (freshness_pred == freshness_labels)
                combined_correct += both_correct.sum().item()
                
                total += fruit_labels.size(0)
        
        return {
            'loss': val_loss / len(self.val_loader),
            'fruit_acc': fruit_correct / total,
            'freshness_acc': freshness_correct / total,
            'combined_acc': combined_correct / total
        }
    
    def _log_metrics(self, epoch, train_loss, val_metrics, stage):
        """Log metrics to console and MLflow"""
        print(f"\nEpoch {epoch+1} (Stage {stage}):")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Fruit Acc: {val_metrics['fruit_acc']:.4f}")
        print(f"  Freshness Acc: {val_metrics['freshness_acc']:.4f}")
        print(f"  Combined Acc: {val_metrics['combined_acc']:.4f}")
        
        # MLflow logging
        mlflow.log_metrics({
            f'stage{stage}_train_loss': train_loss,
            f'stage{stage}_val_loss': val_metrics['loss'],
            f'stage{stage}_fruit_acc': val_metrics['fruit_acc'],
            f'stage{stage}_freshness_acc': val_metrics['freshness_acc'],
            f'stage{stage}_combined_acc': val_metrics['combined_acc']
        }, step=epoch)
        
        # Update history
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_metrics['loss'])
        self.history['val_fruit_acc'].append(val_metrics['fruit_acc'])
        self.history['val_freshness_acc'].append(val_metrics['freshness_acc'])
        self.history['val_combined_acc'].append(val_metrics['combined_acc'])
    
    def _save_checkpoint(self, filename):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }, filename)


# Main training script
def main():
    """Main training function for Azure ML"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--stage1_epochs', type=int, default=10)
    parser.add_argument('--stage2_epochs', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='outputs')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create model
    print("\nInitializing model...")
    model = create_model(num_class=9, pretrained=True)
    
    # Create trainer
    trainer = MultiTaskTrainer(model, train_loader, val_loader, device)
    
    # MLflow tracking
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params({
            'model': 'ResNet50',
            'batch_size': args.batch_size,
            'stage1_epochs': args.stage1_epochs,
            'stage2_epochs': args.stage2_epochs,
            'optimizer': 'AdamW',
            'transfer_learning': 'ImageNet'
        })
        
        # Stage 1: Train heads
        trainer.train_stage1(epochs=args.stage1_epochs, lr=1e-3)
        
        # Stage 2: Fine-tune all
        trainer.train_stage2(epochs=args.stage2_epochs, lr=1e-4)
        
        # Log final model
        mlflow.log_artifact('checkpoint_stage2_best.pth')
        
        print("\nTraining complete!")
        print(f"Best validation accuracy: {trainer.best_val_acc:.4f}")
        
        # Export to ONNX
        export_to_onnx(model, args.output_dir)


def export_to_onnx(model, output_dir):
    """Export model to ONNX format"""
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    model.cpu()
    
    dummy_input = torch.randn(1, 3, 224, 224)
    
    onnx_path = os.path.join(output_dir, 'fruit_classifier.onnx')
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['image'],
        output_names=['fruit_logits', 'freshness_logits', 'confidence'],
        dynamic_axes={
            'image': {0: 'batch_size'},
            'fruit_logits': {0: 'batch_size'},
            'freshness_logits': {0: 'batch_size'},
            'confidence': {0: 'batch_size'}
        }
    )
    
    print(f"\n✓ Model exported to ONNX: {onnx_path}")
    
    # Verify ONNX model
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model verified")
    
    return onnx_path


if __name__ == '__main__':
    main()