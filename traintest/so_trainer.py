import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from feature_cnn_models import SOOpcodeDetailCaptureCNN

class SoTrainer:
    def __init__(self, device, learning_rate=1e-3, margin=0.5, neg_weight=2.0):
        self.device = device
        self.model = SOOpcodeDetailCaptureCNN().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.margin = margin
        self.neg_weight = neg_weight

    def train_epoch(self, train_loader, epoch, show_progress=True):
        self.model.train()
        t_epoch = time.time()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch") if show_progress else train_loader
        
        for x1, x2, y, _, _, _ in pbar:
            x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            
            emb1, emb2 = self.model(x1, x2)
            cos = F.cosine_similarity(emb1, emb2) # [B]
            
            # 损失计算
            pos_loss = (1 - cos[y==1]).mean() if (y==1).any() else 0
            neg_loss = F.relu(cos[y==-1] - self.margin).mean() if (y==-1).any() else 0
            loss = pos_loss + self.neg_weight * neg_loss
            
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            
        epoch_dt = time.time() - t_epoch
        avg_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0
        return avg_loss, epoch_dt

    def validate(self, val_loader, threshold=0.85):
        self.model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for x1, x2, y, _, _, _ in val_loader:
                x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
                e1, e2 = self.model(x1, x2)
                sim = F.cosine_similarity(e1, e2)
                pred = (sim >= threshold).float()
                label_01 = (y == 1).float()
                val_correct += (pred == label_01).sum().item()
                val_total += y.size(0)
        
        val_acc = val_correct / val_total if val_total else 0
        return val_acc

    def run_training(self, train_loader, val_loader, epochs=5, threshold=0.85, show_progress=True):
        print(f"[SoTrainer] Start training for {epochs} epochs...")
        total_time = 0.0
        
        for epoch in range(1, epochs + 1):
            loss, dt = self.train_epoch(train_loader, epoch, show_progress)
            total_time += dt
            val_acc = self.validate(val_loader, threshold)
            print(f"[SO] Epoch {epoch} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f} | Time: {dt:.2f}s")
            
        return total_time

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"[SoTrainer] Model saved to {path}")

    def load_model(self, path):
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(path))
        else:
            self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        self.model.to(self.device)
        print(f"[SoTrainer] Model loaded from {path}")
