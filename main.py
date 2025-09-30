from fastapi import FastAPI, Response, status, File, UploadFile
from fastapi.routing import APIRouter
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel
import asyncio
import os
import warnings

from app.MediaMerger import MediaMergerClass, media_merger
from app.Calculator import calc
import app.SimCLRAnalyse as simclr_module

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
from copy import deepcopy
import PIL
import numpy as np
import urllib.request
from urllib.error import HTTPError

DATASET_PATH = "dataset"
CHECKPOINT_PATH = "saved_models"
NUM_WORKERS = os.cpu_count()
    
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

# Load models at startup to avoid loading them on each request
device = torch.device("cuda" 
                  if torch.cuda.is_available() else "mps" 
                  if torch.mps.is_available() else "cpu")

print("Device:", device)
print("Num Workers:", NUM_WORKERS)

# Suppress PyTorch warnings during model initialization
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

def load_models():
    """Load models with proper error handling and minimal warnings"""
    simclr_model = None
    logreg_model = None
    
    # Suppress warnings during model loading
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Try to load SimCLR model
        try:
            # Create a new SimCLR model
            simclr_model = SimCLR(hidden_dim=128, lr=5e-4, temperature=0.07, weight_decay=1e-4, max_epochs=500)
            
            # Try multiple approaches to load the saved model
            model_loaded = False
            
            # First try: Load with safe globals for weights_only=True
            try:
                with torch.serialization.safe_globals([SimCLR, LogisticRegression]):
                    state_dict = torch.load('./app/simclr_model_256x256.pt', weights_only=True, map_location=device)
                    simclr_model.load_state_dict(state_dict)
                    model_loaded = True
                    print("✓ SimCLR model loaded successfully (safe method)")
            except:
                pass  # Suppress error messages for cleaner output
            
            # Second try: Load with weights_only=False but handle properly
            if not model_loaded:
                try:
                    # Add current module classes to globals for pickle loading
                    import sys
                    current_module = sys.modules[__name__]
                    setattr(current_module, 'SimCLR', SimCLR)
                    setattr(current_module, 'LogisticRegression', LogisticRegression)
                    
                    checkpoint = torch.load('./app/simclr_model_256x256.pt', weights_only=False, map_location=device)
                    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                        simclr_model.load_state_dict(checkpoint['state_dict'])
                    elif hasattr(checkpoint, 'state_dict'):
                        simclr_model.load_state_dict(checkpoint.state_dict())
                    else:
                        simclr_model.load_state_dict(checkpoint)
                    model_loaded = True
                    print("✓ SimCLR model loaded successfully (fallback method)")
                except:
                    pass  # Suppress error messages
            
            if not model_loaded:
                print("⚠ Using fresh SimCLR model (no pre-trained weights found)")
            
            simclr_model.eval()
            
        except Exception as e:
            print(f"⚠ Failed to create SimCLR model: {e}")

        # Try to load LogisticRegression model
        try:
            # Create a new LogisticRegression model
            logreg_model = LogisticRegression(feature_dim=128, num_classes=5, lr=1e-3, weight_decay=1e-4, max_epochs=100)
            
            # Try multiple approaches to load the saved model
            model_loaded = False
            
            # First try: Load with safe globals for weights_only=True
            try:
                with torch.serialization.safe_globals([SimCLR, LogisticRegression]):
                    state_dict = torch.load('./app/logreg_model.pt', weights_only=True, map_location=device)
                    logreg_model.load_state_dict(state_dict)
                    model_loaded = True
                    print("✓ LogisticRegression model loaded successfully (safe method)")
            except:
                pass  # Suppress error messages
            
            # Second try: Load with weights_only=False but handle properly
            if not model_loaded:
                try:
                    checkpoint = torch.load('./app/logreg_model.pt', weights_only=False, map_location=device)
                    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                        logreg_model.load_state_dict(checkpoint['state_dict'])
                    elif hasattr(checkpoint, 'state_dict'):
                        logreg_model.load_state_dict(checkpoint.state_dict())
                    else:
                        logreg_model.load_state_dict(checkpoint)
                    model_loaded = True
                    print("✓ LogisticRegression model loaded successfully (fallback method)")
                except:
                    pass  # Suppress error messages
            
            if not model_loaded:
                print("⚠ Using fresh LogisticRegression model (no pre-trained weights found)")
            
            logreg_model.eval()
                
        except Exception as e:
            print(f"⚠ Failed to create LogisticRegression model: {e}")
    
    return simclr_model, logreg_model

# Load models with proper class context
print("Loading models...")
simclr_model, logreg_model = load_models()

# Pre-compute features at startup
precomputed_data = None
if simclr_model is not None:
    print("Pre-computing features at startup...")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            precomputed_data = simclr_module.precompute_features(simclr_model, simclr_module.DATASET_PATH, simclr_module.class_names)
        
        if precomputed_data[0] is not None:
            print(f"Features pre-computed successfully! Ready to process images.")
        else:
            print("Warning: Feature pre-computation returned empty results.")
            precomputed_data = None
    except Exception as e:
        print(f"Failed to pre-compute features: {e}")
        precomputed_data = None
else:
    print("Warning: SimCLR model not loaded, skipping feature pre-computation.")

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Welcome to the PicTunes API!"}

@app.get("/calc")
def calculate(a: float, b: float, operation: str):
    result = calc(a, b, operation)
    if result is not None:
        return {"result": result}
    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"message": "Invalid operation"})
# to test type curl -X 'GET' 'http://localhost:8001/calc?a=5&b=3&operation=add'
@app.get("/health/")
def health_check():
    return Response(status_code=status.HTTP_200_OK)

@app.get("/dbcon_check/")
def db_connection_check():

    return {"message": "db connection successful"}


@app.get("/upload")
async def img_analysis():
    if simclr_model is None or logreg_model is None:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                          content={"message": "Models not loaded properly"})
    
    if precomputed_data is None:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                          content={"message": "Features not pre-computed properly"})
    
    try:
        # Use the pre-computed features instead of computing them on each request
        all_class_matches, top_10_matches = simclr_module.fast_visualize_prediction(
            image_path=f"/Users/taxihuang/Downloads/IMG_4898.JPG",
            simclr_model=simclr_model,
            logreg_model=logreg_model,
            precomputed_data=precomputed_data,
            class_names=simclr_module.class_names
        )
        return {"all_class_matches": all_class_matches, "top_10_matches": top_10_matches}
    except Exception as e:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                          content={"message": f"Analysis failed: {str(e)}"})
    # return FileResponse(f"uploads/{img.filename}", media_type="video/mp4")

    # curl -X 'GET' 'http://localhost:8001/upload'

@app.post("/media_merger/")
async def merger(img: UploadFile = File(...), aud: UploadFile = File(...)):
    media_merger(img, aud)
    return {"message": "Media merged successfully"} # Response body

