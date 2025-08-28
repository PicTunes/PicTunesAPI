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
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

class SimCLRBackbone(nn.Module):
    """ResNet-18 encoder with projection MLP head."""
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        base = torchvision.models.resnet18(weights=None)
        self.encoder = nn.Sequential(*list(base.children())[:-1])
        feat_dim = base.fc.in_features
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, 4 * hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = torch.flatten(h, 1)
        z = self.projector(h)
        z = F.normalize(z, dim=1)
        return z

class SimCLRService:
    """Load weights, embed images, and top-k retrieval."""
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
        self.db_vectors: Optional[np.ndarray] = None
        self.db_meta: List[Dict] = []
        self.index = None
        self._load_weights(weights_path)

    def _load_weights(self, path: str) -> None:
        if not path or not os.path.exists(path):
            print(f"[SimCLRService] WARNING: weights not found at {path}; using random init")
            return
        print(f"[SimCLRService] Loading weights: {path}")
        state = torch.load(path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            sd = state["state_dict"]
            new_sd = {}
            for k, v in sd.items():
                if k.startswith("convnet.") and not k.startswith("convnet.fc."):
                    new_sd["encoder." + k.replace("convnet.", "")] = v
                elif k.startswith("convnet.fc."):
                    new_sd["projector." + k.replace("convnet.fc.", "")] = v
                elif k.startswith("encoder.") or k.startswith("projector."):
                    new_sd[k] = v
            self.model.load_state_dict(new_sd, strict=False)
        else:
            self.model.load_state_dict(state, strict=False)

    @torch.inference_mode()
    def embed(self, pil_img: Image.Image) -> np.ndarray:
        x = self.transform(pil_img).unsqueeze(0).to(self.device)
        z = self.model(x)
        return z.cpu().numpy()[0].astype("float32")

    def load_database(self, items: List[Tuple[str, str, str]]):
        vecs, meta = [], []
        for url, label, emb_path in items:
            if not emb_path or not os.path.exists(emb_path):
                continue
            v = np.load(emb_path).astype("float32")
            n = np.linalg.norm(v) + 1e-12
            v = v / n
            vecs.append(v)
            meta.append({"url": url, "label": label})
        if not vecs:
            print("[SimCLRService] WARNING: no vectors loaded")
            self.db_vectors, self.db_meta, self.index = None, [], None
            return
        self.db_vectors = np.vstack(vecs).astype("float32")
        self.db_meta = meta
        if HAS_FAISS:
            d = self.db_vectors.shape[1]
            self.index = faiss.IndexFlatIP(d)
            self.index.add(self.db_vectors)
            print(f"[SimCLRService] FAISS index ready: N={len(self.db_meta)} dim={d}")
        else:
            self.index = None
            print(f"[SimCLRService] Using numpy scan: N={len(self.db_meta)}")

    def topk(self, query_vec: np.ndarray, k: int = 3) -> List[Dict]:
        if self.db_vectors is None or len(self.db_meta) == 0:
            return []
        q = query_vec.astype("float32")
        n = np.linalg.norm(q) + 1e-12
        q = q / n
        if self.index is not None:
            D, I = self.index.search(q.reshape(1, -1), k)
            scores, idxs = D[0].tolist(), I[0].tolist()
        else:
            sims = self.db_vectors @ q
            idxs = np.argsort(-sims)[:k]
            scores = sims[idxs].tolist()
        results = []
        for idx, sc in zip(idxs, scores):
            m = self.db_meta[idx]
            results.append({"image_url": m["url"], "label": m.get("label") or "", "similarity": float(max(0.0, min(1.0, sc)))})
        return results