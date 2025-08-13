# app/services/simclr_service.py
import os
from typing import List, Tuple, Optional, Dict
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

try:
    import faiss  # optional acceleration
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False


class SimCLRBackbone(nn.Module):
    """
    ResNet-18 encoder with a projection MLP head as used in SimCLR.
    This matches the structure defined in the notebook where convnet=ResNet18 and fc is replaced by MLP.
    """
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        base = torchvision.models.resnet18(weights=None)
        self.encoder = nn.Sequential(*list(base.children())[:-1])  # up to global pool
        feat_dim = base.fc.in_features
        # Projection head (MLP)
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, 4 * hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)                # (B, C, 1, 1)
        h = torch.flatten(h, 1)            # (B, C)
        z = self.projector(h)              # (B, hidden_dim)
        z = F.normalize(z, dim=1)          # normalized embedding for cosine similarity
        return z


class SimCLRService:
    """
    Self-contained service for:
    - loading SimCLR weights (Lightning .ckpt or plain state_dict)
    - embedding PIL images
    - loading DB embeddings and doing top-k retrieval with FAISS or numpy
    """
    def __init__(self, weights_path: str, device: str = "cpu", hidden_dim: int = 128):
        self.device = torch.device(device)
        self.model = SimCLRBackbone(hidden_dim=hidden_dim).to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.db_vectors: Optional[np.ndarray] = None    # shape (N, D), float32, L2-normalized
        self.db_meta: List[Dict] = []                  # list of {"url": str, "label": str}
        self.index = None
        self._load_weights(weights_path)

    # -------- Weights loading --------
    def _load_weights(self, path: str) -> None:
        if not path or not os.path.exists(path):
            print(f"[SimCLRService] WARNING: weights not found at {path}; using random init")
            return
        print(f"[SimCLRService] Loading weights: {path}")
        state = torch.load(path, map_location=self.device)

        # Lightning .ckpt usually stores "state_dict"
        if isinstance(state, dict) and "state_dict" in state:
            sd = state["state_dict"]
            new_sd = {}
            # Map common prefixes to our backbone
            for k, v in sd.items():
                # Typical patterns seen in notebooks:
                # "convnet.*.weight", "convnet.*.bias" or "encoder.*"
                if k.startswith("convnet."):
                    k2 = k.replace("convnet.", "")
                    # conv layers and projector live in encoder/projector
                    if k2.startswith("fc."):
                        # fc.* are projector layers in this service
                        new_sd["projector." + k2[3:]] = v
                    else:
                        # everything else routes to encoder
                        new_sd["encoder." + k2] = v
                elif k.startswith("encoder."):
                    new_sd[k] = v
                elif k.startswith("model.") or k.startswith("backbone."):
                    # if user saved with custom wrapper
                    k2 = k.split(".", 1)[1]
                    new_sd[k2] = v
                else:
                    # ignore unrelated keys (optimizer, schedulers, etc.)
                    pass
            missing, unexpected = self.model.load_state_dict(new_sd, strict=False)
            print(f"[SimCLRService] Loaded .ckpt; missing={len(missing)} unexpected={len(unexpected)}")
        else:
            # Plain state dict
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            print(f"[SimCLRService] Loaded state_dict; missing={len(missing)} unexpected={len(unexpected)}")

    # -------- Embedding --------
    @torch.inference_mode()
    def embed(self, pil_img: Image.Image) -> np.ndarray:
        """
        Convert a PIL image to a normalized embedding vector.
        """
        x = self.transform(pil_img).unsqueeze(0).to(self.device)
        z = self.model(x)                  # (1, D), already normalized
        return z.cpu().numpy()[0].astype("float32")

    # -------- Database loading --------
    def load_database(self, items: List[Tuple[str, str, str]]):
        """
        Load database embeddings into memory.

        items: list of (url, label, embedding_path)
               embedding_path points to a .npy file storing a float32 vector.
        """
        vecs = []
        meta = []
        for url, label, emb_path in items:
            if not emb_path or not os.path.exists(emb_path):
                continue
            v = np.load(emb_path).astype("float32")
            n = np.linalg.norm(v) + 1e-12
            v = v / n
            vecs.append(v)
            meta.append({"url": url, "label": label})
        if not vecs:
            print("[SimCLRService] WARNING: no vectors loaded into memory")
            self.db_vectors = None
            self.db_meta = []
            self.index = None
            return

        self.db_vectors = np.vstack(vecs).astype("float32")
        self.db_meta = meta
        if HAS_FAISS:
            d = self.db_vectors.shape[1]
            self.index = faiss.IndexFlatIP(d)  # cosine via inner product if normalized
            self.index.add(self.db_vectors)
            print(f"[SimCLRService] FAISS index ready: N={len(self.db_meta)} dim={d}")
        else:
            self.index = None
            print(f"[SimCLRService] Using numpy scan: N={len(self.db_meta)}")

    # -------- Retrieval --------
    def topk(self, query_vec: np.ndarray, k: int = 3) -> List[Dict]:
        """
        Return top-k nearest neighbors as a list of dicts:
        {"image_url": str, "label": str, "similarity": float in [0,1]}
        """
        if self.db_vectors is None or len(self.db_meta) == 0:
            return []

        q = query_vec.astype("float32")
        n = np.linalg.norm(q) + 1e-12
        q = q / n

        if HAS_FAISS and self.index is not None:
            D, I = self.index.search(q.reshape(1, -1), k)
            scores = D[0].tolist()
            idxs = I[0].tolist()
        else:
            sims = self.db_vectors @ q  # cosine if normalized
            idxs = np.argsort(-sims)[:k]
            scores = sims[idxs].tolist()

        results = []
        for idx, sc in zip(idxs, scores):
            m = self.db_meta[idx]
            results.append({
                "image_url": m["url"],
                "label": m.get("label") or "",
                "similarity": float(max(0.0, min(1.0, sc)))
            })
        return results
