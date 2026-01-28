#training script

import torch
import torch.nn as nn
from torchvision import models


"""
Multi-task Classification architecture
Base Model: ResNet50
Fruit/vegetable identification
Fresh/Rotten Classsification
Probable Mold estimation
"""

class FreshResNet(nn.Module):
    def __init__(self, num_class=9, num_fresh=2, pretrained=True):
        super(FreshResNet, self).__init__()

        resnet = models.resnet50(pretrained=pretrained)  #Loading pretrained model

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        num_features = 2048

        self.freeze_backbone()   # freeze backbone

        #Task 1 : produce class
        self.fruit_classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_class)
        )

        #Task 2: Detect Freshness
        self.freshness_classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_fresh)
        )

        #Task 3: Freshness Level
        self.confidence_regressor = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Outputs 0-1
        )


    def freeze_backbone(self):
        """Freeze backbone for Stage 1 training"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("✓ Backbone frozen (Stage 1: Train heads only)")
    
    def unfreeze_backbone(self, unfreeze_from_layer=6):
        """
        Unfreeze backbone from specified layer for Stage 2 fine-tuning
        ResNet50 has 4 main blocks (layer1-4), we typically unfreeze layer3-4
        """
        layers = list(self.backbone.children())

        # Unfreeze from layer index onwards
        for layer in layers[unfreeze_from_layer:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"✓ Backbone partially unfrozen (Stage 2: Fine-tuning)")
        print(f"  Trainable parameters: {trainable:,}")

    
    def forward(self, x):
        # Extract features with ResNet backbone
        features = self.backbone(x)
        features = features.view(features.size(0), -1)  # Flatten: [B, 2048]
        
        # Multi-task predictions
        fruit_logits = self.fruit_classifier(features)
        freshness_logits = self.freshness_classifier(features)
        mold_confidence = self.confidence_regressor(features)
        
        return fruit_logits, freshness_logits, mold_confidence
    

    def predict_moldness_level(self, freshness_class, freshness_prob, confidence_score):
        """
        Estimate moldness level (0-5) from classification + confidence
        
        Logic:
        - Fresh predictions with high confidence → Level 0-1
        - Fresh predictions with low confidence → Level 2
        - Rotten predictions with varying confidence → Level 3-5
        
        Args:
            freshness_class: 0 (fresh) or 1 (rotten)
            freshness_prob: P(fresh) from softmax
            confidence_score: Regression output (0-1)
        
        Returns:
            moldness_level: 0 (perfect) to 5 (severe)
        """
        levels = []
        
        for cls, prob, conf in zip(freshness_class, freshness_prob, confidence_score):
            if cls == 0:  # Predicted fresh
                if prob > 0.95 and conf > 0.9:
                    level = 'Very Fresh / Sehr Frisch'  # Perfect
                elif prob > 0.85 and conf > 0.75:
                    level = 'Fresh / Frisch'  # Minor spots
                elif prob > 0.70:
                    level = 'Good / Gut'  # Early mold
                else:
                    level = 'OK / Geht'  # Borderline
            else:  # Predicted rotten
                if prob < 0.60 and conf < 0.4:
                    level = 'Slight mold / Leichter Schimmel'  # Moderate
                elif prob < 0.40 and conf < 0.25:
                    level = 'Mold / Schimmel'  # Heavy mold
                else:
                    level = 'Rotten /  Verrotten'  # Severe
            
            levels.append(level)
        
        return torch.tensor(levels)
    
    def predict(self, images, class_names=None):
        """
        Complete prediction pipeline returning item type, freshness, and mold level
        
        Args:
            images: Input tensor of shape [B, 3, H, W]
            class_names: List of class names (e.g., ['apple', 'banana', ...])
        
        Returns:
            Dictionary with predictions:
            - 'item': predicted item type
            - 'item_confidence': confidence for item prediction
            - 'freshness': 'Fresh' or 'Rotten'
            - 'freshness_confidence': confidence for freshness prediction
            - 'mold_level': mold/quality level with description
        """
        if class_names is None:
            class_names = ['apple', 'banana', 'bitterground', 'capsicum', 'cucumber', 
                          'okra', 'orange', 'potato', 'tomato']
        
        with torch.no_grad():
            fruit_logits, freshness_logits, mold_conf = self.forward(images)
        
        # Get item predictions
        fruit_probs = torch.softmax(fruit_logits, dim=1)
        item_preds = torch.argmax(fruit_logits, dim=1)
        item_confidences = torch.max(fruit_probs, dim=1)[0]
        item_names = [class_names[idx] for idx in item_preds.cpu().numpy()]
        
        # Get freshness predictions
        freshness_probs = torch.softmax(freshness_logits, dim=1)
        freshness_preds = torch.argmax(freshness_logits, dim=1)
        freshness_confidences = torch.max(freshness_probs, dim=1)[0]
        freshness_names = ['Fresh' if pred == 0 else 'Rotten' for pred in freshness_preds.cpu().numpy()]
        
        # Get mold level
        freshness_probs_class0 = freshness_probs[:, 0]
        mold_levels = self.predict_moldness_level(freshness_preds, freshness_probs_class0, mold_conf.squeeze())
        
        return {
            'item': item_names,
            'item_confidence': item_confidences.cpu().numpy(),
            'freshness': freshness_names,
            'freshness_confidence': freshness_confidences.cpu().numpy(),
            'mold_level': mold_levels,
            'mold_confidence': mold_conf.cpu().numpy()
        }
    

# Model factory
def create_model(num_class=9, pretrained=True):
    """Factory function to create model"""
    return FreshResNet(
        num_class=num_class,
        num_fresh=2,
        pretrained=pretrained
    )