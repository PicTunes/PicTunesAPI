# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

# TorchVision
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

# Others
import os
import re
import warnings
from copy import deepcopy
import PIL
import numpy as np
import urllib.request
from urllib.error import HTTPError

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Path Configurations
DATASET_PATH = "dataset"
CHECKPOINT_PATH = "saved_models"
NUM_WORKERS = os.cpu_count()

# Device configuration
device = torch.device("cuda" 
                     if torch.cuda.is_available() else "mps" 
                     if torch.mps.is_available() else "cpu")

# Seed Settings
pl.seed_everything(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class ContrastiveTransformations(object):
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]


contrast_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(size=192),
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.5, 
                               contrast=0.5, 
                               saturation=0.5, 
                               hue=0.1)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(kernel_size=23),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def create_data_loaders(data_dir, batch_size, train_ratio=0.8, num_workers=2):
    """
    Creates training and validation data loaders.
    """
    # Load the full dataset with contrastive transformations
    full_dataset = ImageFolder(
        root=data_dir,
        transform=ContrastiveTransformations(contrast_transforms, n_views=2)
    )

    print(f"Full dataset size: {len(full_dataset)} images")
    print(f"Classes: {full_dataset.classes}")

    # Calculate split sizes
    dataset_size = len(full_dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size

    # Split dataset randomly
    train_dataset, val_dataset = data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create data loaders
    train_loader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        drop_last=True, pin_memory=True, num_workers=num_workers
    )
    val_loader = data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        drop_last=False, pin_memory=True, num_workers=num_workers
    )   

    return train_loader, val_loader, full_dataset.classes

# Load dataset
train_loader, val_loader, class_names = create_data_loaders(
    DATASET_PATH, batch_size=256, num_workers=NUM_WORKERS
)

def precompute_features(simclr_model, dataset_path, class_names):
    """
    Precompute features for all images in the dataset.
    
    Args:
        simclr_model: The SimCLR model for feature extraction
        dataset_path: Path to the dataset directory
        class_names: List of class names to process
    
    Returns:
        Tuple of (features, paths, classes) or (None, [], []) if failed
    """
    try:
        # Use the full SimCLR convnet (with projection layer) for feature extraction
        feature_extractor = deepcopy(simclr_model.convnet)
        feature_extractor.eval().to(device)

        eval_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        all_features = []
        all_paths = []
        all_classes = []

        print(f"Starting feature precomputation for {len(class_names)} classes...")

        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(dataset_path, class_name)
            if not os.path.isdir(class_dir):
                print(f"Warning: Class directory '{class_dir}' not found, skipping...")
                continue

            # Get all image files (not just numeric ones)
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
            all_images = [img for img in os.listdir(class_dir) 
                         if os.path.splitext(img.lower())[1] in image_extensions]
            
            # Prioritize numeric images but include all
            numeric_images = [img for img in all_images if re.match(r'^\d+', img)]
            non_numeric_images = [img for img in all_images if not re.match(r'^\d+', img)]
            images_to_process = numeric_images + non_numeric_images
            
            print(f"Class '{class_name}': Processing {len(images_to_process)} images...")

            batch_size = 64
            processed_count = 0
            
            for i in range(0, len(images_to_process), batch_size):
                batch_images = images_to_process[i:i+batch_size]
                batch_tensors = []
                batch_paths = []

                for img_name in batch_images:
                    img_path = os.path.join(class_dir, img_name)
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            img = PIL.Image.open(img_path).convert('RGB')
                            tensor = eval_transform(img).unsqueeze(0)
                            batch_tensors.append(tensor)
                            batch_paths.append(img_path)
                            processed_count += 1
                    except Exception as e:
                        print(f"Warning: Error processing image {img_path}: {e}")
                        continue

                if batch_tensors:
                    try:
                        batch_tensors = torch.cat(batch_tensors).to(device)
                        with torch.no_grad():
                            batch_features = feature_extractor(batch_tensors).cpu()

                        all_features.append(batch_features)
                        all_paths.extend(batch_paths)
                        all_classes.extend([class_name] * len(batch_paths))
                    except Exception as e:
                        print(f"Warning: Error processing batch for class {class_name}: {e}")
                        continue
                        
            print(f"Class '{class_name}': Successfully processed {processed_count} images")
                
        if all_features:
            all_features = torch.cat(all_features, dim=0)
            print(f"Successfully precomputed features for {len(all_features)} images across {len(set(all_classes))} classes.")
            return all_features, all_paths, all_classes
        else:
            print("Error: No features were precomputed.")
            return None, [], []
    
    except Exception as e:
        print(f"Error in precompute_features: {e}")
        return None, [], []
    
def fast_visualize_prediction(image_path, simclr_model, logreg_model, precomputed_data, class_names):
    precomputed_features, precomputed_paths, precomputed_classes = precomputed_data

    img = PIL.Image.open(image_path).convert('RGB')
    eval_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    input_tensor = eval_transform(img).unsqueeze(0).to(device)

    # Use the full SimCLR convnet (with projection layer) for feature extraction
    feature_extractor = deepcopy(simclr_model.convnet)
    feature_extractor.eval().to(device)
    logreg_model = logreg_model.to(device)

    with torch.no_grad():
        input_features = feature_extractor(input_tensor)
        preds = logreg_model.model(input_features)
        pred_class_idx = preds.argmax(dim=-1).item()
        pred_class_name = class_names[pred_class_idx]

    try:
        true_class_name = image_path.split(os.sep)[-2]
    except:
        true_class_name = "Unknown"

    input_features_cpu = input_features.cpu()
    all_similarities = F.cosine_similarity(input_features_cpu, precomputed_features)

    similarity_data = []
    for i, (sim, path, cls) in enumerate(zip(all_similarities, precomputed_paths, precomputed_classes)):
        similarity_data.append((sim.item(), path, cls))

    all_top_matches = sorted(similarity_data, reverse=True, key=lambda x: x[0])[:10]

    pred_class_similarities = [item for item in similarity_data if item[2] == pred_class_name]

    pred_class_similarities.sort(reverse=True, key=lambda x: x[0])
    top_10_matches = pred_class_similarities[:10]

    print(f"Ground Truth: {true_class_name}\nPredicted Class: {pred_class_name}\n")

    print("Top 10 matches from the predicted class:\n")
    # for i, (sim, path, cls) in enumerate(all_top_matches, 1):
    #     print(f"{i}. File: {os.path.basename(path)}, Class: {cls}, Similarity: {sim:.3f}")
    print(all_top_matches)

    if top_10_matches:
        print(f"\nPredicted class '{pred_class_name}' Top10:")
        # for i, (sim, path, cls) in enumerate(top_10_matches, 1):
        #     print(f"{i}. File: {os.path.basename(path)}, Similarity: {sim:.3f}")
        print(top_10_matches)
        best_sim, best_match_path, best_match_class = all_top_matches[0]

        # best_match_img = PIL.Image.open(best_match_path).convert('RGB')
        # best_match_tensor = eval_transform(best_match_img)
    else:
        print(f"No matches found for class {pred_class_name}.")

    return all_top_matches, top_10_matches

# simclr_model = torch.load('./app/simclr_model_256x256.pt', weights_only=False)
# simclr_model.eval()


# Deprecated: Use fast_visualize_prediction with pre-computed data instead
# This function recomputes features every time, which is inefficient
def simclr_img_analysis_deprecated(image_path, simclr_model, logreg_model):
    """
    DEPRECATED: This function recomputes features every time it's called.
    Use fast_visualize_prediction with pre-computed data instead for better performance.
    """
    print("WARNING: Using deprecated function that recomputes features. Use fast_visualize_prediction instead.")
    precomputed_data = precompute_features(simclr_model, DATASET_PATH, class_names)
    all_class_matches, top_10_matches = fast_visualize_prediction(
        image_path=image_path,
        simclr_model=simclr_model,
        logreg_model=logreg_model,
        precomputed_data=precomputed_data,
        class_names=class_names
    )
    return all_class_matches, top_10_matches

def simclr_img_analysis(image_path, simclr_model, logreg_model, precomputed_data=None, class_names=None):
    """
    Efficient image analysis using pre-computed features.
    
    Args:
        image_path: Path to the image to analyze
        simclr_model: Pre-loaded SimCLR model
        logreg_model: Pre-loaded LogisticRegression model
        precomputed_data: Pre-computed features (features, paths, classes)
        class_names: List of class names
    
    Returns:
        Tuple of (all_class_matches, top_10_matches)
    """
    if precomputed_data is None:
        print("WARNING: No pre-computed data provided. Computing features on-the-fly (slower).")
        precomputed_data = precompute_features(simclr_model, DATASET_PATH, class_names or [])
    
    if class_names is None:
        # Try to get class names from precomputed data or use default
        try:
            from torchvision.datasets import ImageFolder
            dataset = ImageFolder(root=DATASET_PATH)
            class_names = dataset.classes
        except:
            class_names = ['Architecture', 'Food', 'Landscape', 'Outfit', 'Sports']  # Default fallback
    
    all_class_matches, top_10_matches = fast_visualize_prediction(
        image_path=image_path,
        simclr_model=simclr_model,
        logreg_model=logreg_model,
        precomputed_data=precomputed_data,
        class_names=class_names
    )
    return all_class_matches, top_10_matches
