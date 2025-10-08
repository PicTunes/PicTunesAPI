"""
SimCLR Analysis Module
Handles image classification and similarity search using pre-trained SimCLR and Logistic Regression models
"""

import os
import re
from copy import deepcopy
from typing import Tuple, List, Dict
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision
import pytorch_lightning as pl
import PIL
import numpy as np

warnings.filterwarnings('ignore')

# Path configurations
DATASET_PATH = "./dataset/"
MODEL_PATH = "./app/"

# Device configuration
device = torch.device("mps") if torch.mps.is_available() else torch.device("cpu")
NUM_WORKERS = os.cpu_count() if os.cpu_count() else 2

print(f"[SimCLR] Device: {device}")
print(f"[SimCLR] NUM_WORKERS: {NUM_WORKERS}")


class SimCLR(pl.LightningModule):
    """
    SimCLR model for contrastive learning
    """
    def __init__(self, hidden_dim, lr, temperature, weight_decay, max_epochs=500):
        super().__init__()
        self.save_hyperparameters()
        assert self.hparams.temperature > 0.0, 'The temperature must be a positive float!'

        # Base model f(.) - ResNet-50
        self.convnet = torchvision.models.resnet50(weights=None)
    
        # The MLP for g(.) consists of Linear->ReLU->Linear
        self.convnet.fc = nn.Sequential(
            nn.Linear(self.convnet.fc.in_features, 4*hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(4*hidden_dim, hidden_dim)
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
            eta_min=self.hparams.lr/50
        )
        return [optimizer], [lr_scheduler]

    def info_nce_loss(self, batch, mode='train'):
        imgs, _ = batch
        imgs = torch.cat(imgs, dim=0)

        # Encode all images
        feats = self.convnet(imgs)

        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(feats[:,None,:], feats[None,:,:], dim=-1)

        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)

        # Find positive example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)

        # InfoNCE loss
        cos_sim = cos_sim / self.hparams.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        return nll

    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode='train')

    def validation_step(self, batch, batch_idx):
        self.info_nce_loss(batch, mode='val')


class LogisticRegression(pl.LightningModule):
    """
    Logistic regression for classification
    """
    def __init__(self, feature_dim, num_classes, lr, weight_decay, max_epochs=100):
        super().__init__()
        self.save_hyperparameters()
        # Mapping from representation h to classes
        self.model = nn.Linear(feature_dim, num_classes)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(self.hparams.max_epochs*0.6), int(self.hparams.max_epochs*0.8)],
            gamma=0.1
        )
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode='train'):
        feats, labels = batch
        preds = self.model(feats)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        return loss

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode='train')

    def validation_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode='val')

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode='test')


def precompute_features(simclr_model, dataset_path, class_names):
    """
    Precompute features for all numeric-named images in the dataset
    
    Args:
        simclr_model: Trained SimCLR model
        dataset_path: Path to dataset directory
        class_names: List of class names
        
    Returns:
        Tuple of (features tensor, image paths list, class names list)
    """
    print("[SimCLR] Precomputing features for numeric images...")
    
    # Prepare feature extractor
    feature_extractor = deepcopy(simclr_model.convnet)
    feature_extractor.fc = nn.Identity()
    feature_extractor.eval().to(device)
    
    # Image transformation for evaluation
    eval_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Storage for features, paths, and classes
    all_features = []
    all_paths = []
    all_classes = []
    
    # Process each class directory
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        # Find all numeric-named images
        numeric_images = [
            img for img in os.listdir(class_dir) 
            if re.match(r'^\d+', img) and img.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        print(f"[SimCLR] Class '{class_name}': Found {len(numeric_images)} numeric images")
        
        # Process in batches
        batch_size = 64
        for i in range(0, len(numeric_images), batch_size):
            batch_images = numeric_images[i:i+batch_size]
            batch_tensors = []
            batch_paths = []
            
            # Load and transform images
            for img_name in batch_images:
                img_path = os.path.join(class_dir, img_name)
                try:
                    img = PIL.Image.open(img_path).convert("RGB")
                    tensor = eval_transform(img).unsqueeze(0)
                    batch_tensors.append(tensor)
                    batch_paths.append(img_path)
                except Exception as e:
                    print(f"[SimCLR] Error processing {img_path}: {e}")
            
            if batch_tensors:
                # Extract features in batch
                batch_tensors = torch.cat(batch_tensors).to(device)
                with torch.no_grad():
                    batch_features = feature_extractor(batch_tensors).cpu()
                
                # Save features and paths
                all_features.append(batch_features)
                all_paths.extend(batch_paths)
                all_classes.extend([class_name] * len(batch_paths))
    
    # Combine all features
    if all_features:
        all_features = torch.cat(all_features, dim=0)
        print(f"[SimCLR] Completed feature precomputation for {len(all_paths)} images")
        return all_features, all_paths, all_classes
    else:
        print("[SimCLR] No numeric images found")
        return None, [], []


def fast_visualize_prediction(
    image_path: str,
    simclr_model,
    logreg_model,
    precomputed_data: Tuple,
    class_names: List[str]
) -> Tuple[Dict[str, List], List[Dict]]:
    """
    Classify image and find similar images using precomputed features
    
    Args:
        image_path: Path to input image
        simclr_model: Trained SimCLR model
        logreg_model: Trained logistic regression model
        precomputed_data: Tuple of (features, paths, classes)
        class_names: List of class names
        
    Returns:
        Tuple of (all_class_matches dict, top_10_matches list)
        - all_class_matches: Dict with keys 'overall' and 'predicted_class', each containing list of match dicts
        - top_10_matches: List of top 10 match dicts from predicted class
    """
    # Unpack precomputed data
    precomputed_features, precomputed_paths, precomputed_classes = precomputed_data
    
    # Load and transform input image
    img = PIL.Image.open(image_path).convert("RGB")
    eval_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    input_tensor = eval_transform(img).unsqueeze(0).to(device)
    
    # Prepare models
    feature_extractor = deepcopy(simclr_model.convnet)
    feature_extractor.fc = nn.Identity()
    feature_extractor.eval().to(device)
    logreg_model = logreg_model.to(device)
    
    with torch.no_grad():
        # Extract input image features
        input_features = feature_extractor(input_tensor)
        
        # Predict class
        preds = logreg_model.model(input_features)
        pred_class_idx = preds.argmax(dim=-1).item()
        pred_class_name = class_names[pred_class_idx]
        pred_probabilities = F.softmax(preds, dim=-1)[0]
    
    # Calculate similarities with all precomputed features
    input_features_cpu = input_features.cpu()
    all_similarities = F.cosine_similarity(input_features_cpu, precomputed_features)
    
    # Create similarity data list
    similarity_data = []
    for i, (sim, path, cls) in enumerate(zip(all_similarities, precomputed_paths, precomputed_classes)):
        filename = os.path.basename(path)
        # Extract only numeric part from filename
        numeric_name = re.match(r'^(\d+)', filename)
        if numeric_name:
            filename = numeric_name.group(1)
        
        similarity_data.append({
            'similarity': float(sim.item()),
            'filename': filename,
            'class': cls,
            'full_path': path
        })
    
    # Sort by similarity and get top 10 overall matches
    all_top_matches = sorted(similarity_data, key=lambda x: x['similarity'], reverse=True)[:10]
    
    # Filter for predicted class and get top 10
    pred_class_similarities = [item for item in similarity_data if item['class'] == pred_class_name]
    pred_class_similarities.sort(key=lambda x: x['similarity'], reverse=True)
    top_10_matches = pred_class_similarities[:10]
    
    # Prepare response
    all_class_matches = {
        'predicted_class': pred_class_name,
        'confidence': float(pred_probabilities[pred_class_idx].item()),
        'all_class_probabilities': {
            class_name: float(prob.item()) 
            for class_name, prob in zip(class_names, pred_probabilities)
        },
        'overall_top_10': all_top_matches,
        'predicted_class_top_10': top_10_matches
    }
    
    print(f"[SimCLR] Predicted class: {pred_class_name} (confidence: {all_class_matches['confidence']:.3f})")
    print(f"[SimCLR] Found {len(top_10_matches)} matches in predicted class")
    print(f"[SimCLR] Top overall similarity: {all_top_matches[0]['similarity']:.3f}")
    
    return all_class_matches, top_10_matches


# Load models and precompute features at module initialization
print("[SimCLR] Loading models...")

# Workaround for loading models saved with __main__ context
# We need to register our classes in the __main__ namespace
import sys
import __main__
__main__.SimCLR = SimCLR
__main__.LogisticRegression = LogisticRegression

try:
    # Load SimCLR model
    simclr_model_path = os.path.join(MODEL_PATH, "simclr_model_256x256.pt")
    simclr_model = torch.load(simclr_model_path, map_location=device, weights_only=False)
    simclr_model.eval()
    print("[SimCLR] SimCLR model loaded successfully")
except Exception as e:
    print(f"[SimCLR] Error loading SimCLR model: {e}")
    simclr_model = None

try:
    # Load Logistic Regression model
    logreg_model_path = os.path.join(MODEL_PATH, "logreg_model.pt")
    logreg_model = torch.load(logreg_model_path, map_location=device, weights_only=False)
    logreg_model.eval()
    print("[SimCLR] Logistic Regression model loaded successfully")
except Exception as e:
    print(f"[SimCLR] Error loading Logistic Regression model: {e}")
    logreg_model = None

# Define class names
class_names = ['Architecture', 'Food', 'Landscape', 'Outfit', 'Sports']

# Precompute features for all numeric images
precomputed_data = None
if simclr_model is not None:
    try:
        precomputed_data = precompute_features(simclr_model, DATASET_PATH, class_names)
        print("[SimCLR] Feature precomputation completed")
    except Exception as e:
        print(f"[SimCLR] Error during feature precomputation: {e}")
        precomputed_data = None
else:
    print("[SimCLR] Skipping feature precomputation (SimCLR model not loaded)")

print("[SimCLR] Module initialization complete")

