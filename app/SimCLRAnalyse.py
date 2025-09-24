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
from pytorch_lighting import loggers as pl_loggers

# Others
import os
from copy import deepcopy
import PIL
import numpy as np
import urllib.request
from urllib.error import HTTPError

# Path Configurations
DATASET_PATH = "../dataset/"
CHECKPOINT_PATH = "../saved_models/"
NUM_WORKERS = os.cpu_count()

# Seed Settings
pl.seed_everything(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" 
                      if torch.cuda.is_available() else "mps" 
                      if torch.mps.is_available() else "cpu")

print("Device:", device)
print("Num Workers:", NUM_WORKERS)

pretrained_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/tutorial17/"
pretrained_files = ["SimCLR.ckpt", "ResNet.ckpt",
                    "tensorboards/SimCLR/events.out.tfevents.SimCLR",
                    "tensorboards/classification/ResNet/events.out.tfevents.ResNet"]
pretrained_files += [f"LogisticRegression_{size}.ckpt" for size in [10, 20, 50, 100, 200, 500]]
os.makedirs(CHECKPOINT_PATH, exist_ok=True)
for file_name in pretrained_files:
    file_path = os.path.join(CHECKPOINT_PATH, file_name)
    if "/" in file_name:
        os.makedirs(file_path.rsplit("/", 1)[0], exist_ok=True)
    if not os.path.isfile(file_path):
        file_url = pretrained_url + file_name
        print(f"Downloading {file_name}...")
        try:
            urllib.request.urlretrieve(file_url, file_path)
        except HTTPError as e:
            print(f"Failed to download {file_name}: {e}")

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

class SimCLR(pl.LightningModule):
    def __init__(self, hidden_dim, lr, temperature, weight_decay, max_epochs):
        super().__init__()
        self.save_hyperparameters()
        assert self.hparams.temperature > 0.0, "The temperature must be a positive float!"

        # Base model f(.) - ResNet50
        self.convnet = torchvision.models.resnet50(weights=None)

        # MLP g(.) Linear->ReLU->Linear
        self.convnet.fc = nn.Sequential(
            nn.Linear(self.convnet.fc.in_features, 4*hidden_dim),
            nn.ReLU(),
            nn.Linear(4*hidden_dim, hidden_dim)
        )

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=self.hparams.max_epochs,
                                                            eta_min=self.hparams.lr/50)
        return [optimizer], [lr_scheduler]
    
    def info_nce_loss(self, batch, mode='train'):
        imgs, _ = batch
        imgs = torch.cat(imgs, dim=0)
        
        # Encode images
        feats = self.convnet(imgs)

        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(feats[:,None,:], feats[None,:,:], dim=-1)

        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)

        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)

        # InfoNCE loss
        cos_sim = cos_sim / self.hparams.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        # Logging loss
        self.log(mode+"_loss", nll)

        # Get ranking position of positive example
        comb_sim = torch.cat([cos_sim[pos_mask][:,None], 
                                cos_sim.masked_fill(pos_mask, -9e15)], dim=-1)
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)

        # Logging ranking metrics
        self.log(mode+"_acc_top1", (sim_argsort==0).float().mean())
        self.log(mode+"_acc_top5", (sim_argsort<5).float().mean())
        self.log(mode+"_acc_mean_pos", 1+sim_argsort.float().mean())

        return nll
    
    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode='train')
    
    def validation_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode='val')
        
def train_simclr(batch_size, max_epochs=500, **kwargs):
    """
    This function has to run roughly 50min
    TODO: Look for alternatives to store model for accomplishing the classfication task
    """
    # create data loaders
    train_loader, val_loader = create_data_loaders(
        DATASET_PATH, batch_size=batch_size, num_workers=0
    )

    # Setup TensorBoard logging
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=os.path.join(CHECKPOINT_PATH, 'tensorboards'),
        name='SimCLR.ckpt'
    )

    # Configure trainer 
    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, 'SimCLR.ckpt'),
        accelerator="gpu" if torch.cuda.is_available() else "mps" 
                        if torch.mps.is_available() else "cpu",
        devices=1,
        max_epochs=max_epochs,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode='max', monitor='val_acc_top5'),
            LearningRateMonitor('epoch')
        ],
        logger=tb_logger
    )

    # Initialize and train model
    pl.seed_everything(42)
    model = SimCLR(max_epochs=max_epochs, **kwargs)
    trainer.fit(model, train_loader, val_loader)

    # Load best checkpoint
    model = SimCLR.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    return model, trainer

simclr_model, simclr_trainer = train_simclr(
    batch_size=256,
    hidden_dim=128,
    lr=3e-4,
    temperature=0.07,
    weight_decay=1e-4,
    max_epochs=100
) # took around 50 mins to train

simclr_model.to(device)
simclr_model.eval()
print("SimCLR model trained and loaded.")
# Example usage
# for batch in train_loader:
#     imgs, _ = batch
#     imgs = torch.cat(imgs, dim=0).to(device)
#     with torch.no_grad():
#         feats = simclr_model.convnet(imgs)
#     print("Feature shape:", feats.shape)
#     break

simclr_trainer.save_checkpoint(os.path.join(CHECKPOINT_PATH, "SimCLR.ckpt"))
print("SimCLR model checkpoint saved.")

class LogisticRegression(pl.LightningModule):
    def __init__(self, feature_dim, num_classes, lr, weight_decay, max_epochs):
        """
        Logistic regression implementation
        """
        super().__init__()
        self.save_hyperparameters()
        # Mapping from representation h to classes
        self.model = nn.Linear(feature_dim, num_classes)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                      milestones=[int(self.hparams.max_epochs*0.6),
                                                                  int(self.hparams.max_epochs*0.8)],
                                                      gamma=0.1)
        return [optimizer], [lr_scheduler]
    
    def _calculate_loss(self, batch, mode='train'):
        feats, labels = batch
        preds = self.model(feats)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        self.log(mode + "_loss", loss)
        self.log(mode + "_acc", acc)
        return loss
    
    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode='train')
    
    def validation_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode='val')

    def test_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode='test')
    
    
    @torch.no_grad()
    def prepare_data_features(model, dataset): # second arg not used
        """
        Prepare features for evaluation
        """
        # Prepare model - remove projection head g(.)
        network = deepcopy(model.convnet)
        network.fc = nn.Identity()
        network.eval()
        network.to(device)

        # Create evaluation transforms (no augmentations)
        img_transforms = transforms.Compose([
            transforms.Resize(256, 256),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Create dataset with evaluation transforms
        eval_dataset = ImageFolder(root=DATASET_PATH, transforms=img_transforms)

        # Encode all images
        data_loader = data.DataLoader(eval_dataset, batch_size=64,
                                      num_workers=NUM_WORKERS, shuffle=False, drop_last=False)
        feats, labels = [], []
        for batch_imgs, batch_labels in data_loader:
            batch_imgs = batch_imgs.to(device)
            batch_feats = network(batch_imgs)
            feats.append(batch_feats.detach().cpu())
            labels.append(batch_labels)

        feats = torch.cat(feats, dim=0)
        labels = torch.cat(labels, dim=0)

        # Sort images by labels
        labels, idxs = labels.sort()
        feats = feats[idxs]

        return data.TensorDataset(feats, labels)
    
    train_feats_simclr = prepare_data_features(simclr_model, DATASET_PATH) # Run Time: ~1min
    print(f"Features extracted: {train_feats_simclr.tensors[0].shape}")

    def train_logreg(batch_size, train_feats_data, test_feats_data, model_suffix, max_epochs=100, **kwargs):
        """
        Train Logistic Regression
        """
        trainer = pl.Trainer(
            default_root_dir=os.path.join(CHECKPOINT_PATH, "LogisticRegression"),
            accelerator="gpu" if torch.cuda.is_available() else "mps" 
                        if torch.mps.is_available() else "cpu",
            devices=1,
            max_epochs=max_epochs,
            callbacks=[
                ModelCheckpoint(save_weights_only=True, mode='max', monitor='val_acc'),
                LearningRateMonitor('epoch')
            ],
            enable_progress_bar=False,
            check_val_every_n_epoch=10
        )
        trainer.logger._default_hp_metric = None

        # Data Loaders
        train_loader = data.DataLoader(train_feats_data, batch_size=batch_size, shuffle=True,
                                        drop_last=True, pin_memory=True, num_workers=0)
        test_loader = data.DataLoader(test_feats_data, batch_size=batch_size, shuffle=False,
                                        drop_last=False, pin_memory=True, num_workers=0)
        
        # Check whether pretrained model exists
        pretrained_filename = os.path.join(CHECKPOINT_PATH, f"LogisticRegression_{model_suffix}.ckpt")
        if os.path.isfile(pretrained_filename):
            print(f"Found pretrained model: {pretrained_filename}. Loading...")
            model = LogisticRegression.load_from_checkpoint(pretrained_filename)
        else:
            pl.seed_everything(42)
            model = LogisticRegression(**kwargs)
            trainer.fit(model, train_loader, test_loader)