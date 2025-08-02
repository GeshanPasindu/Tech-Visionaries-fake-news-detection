import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import os
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score,confusion_matrix, classification_report

from config import *
from data_loader import get_data_loaders
from model import LRATF
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F



class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p = torch.sigmoid(inputs)
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = bce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


def train_one_epoch(model, data_loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    all_labels = []
    all_preds = []

    for frames, landmarks, motion_vectors, labels in tqdm(data_loader, desc="Training"):
        frames = frames.to(device)
        landmarks = landmarks.to(device)
        motion_vectors = motion_vectors.to(device)
        labels = labels.to(device).unsqueeze(1)

        outputs = model(frames, landmarks, motion_vectors)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        total_loss += loss.item()
        
        preds = torch.sigmoid(outputs).detach().cpu().numpy()
        all_preds.extend(preds.flatten())
        all_labels.extend(labels.cpu().numpy().flatten())

    avg_loss = total_loss / len(data_loader)
    
    if len(set(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_preds)
    else:
        auc = 0.5
    
    binary_preds = [1 if p > 0.5 else 0 for p in all_preds]
    acc = accuracy_score(all_labels, binary_preds)
    # cm = confusion_matrix(all_labels, binary_preds)
    
    return avg_loss, auc, acc

def validate_one_epoch(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for frames, landmarks, motion_vectors, labels in tqdm(data_loader, desc="Validating"):
            frames = frames.to(device)
            landmarks = landmarks.to(device)
            motion_vectors = motion_vectors.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(frames, landmarks, motion_vectors)
            loss = loss_fn(outputs, labels)
            
            total_loss += loss.item()
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    avg_loss = total_loss / len(data_loader)

    if len(set(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_preds)
    else:
        auc = 0.5

    binary_preds = [1 if p > 0.5 else 0 for p in all_preds]
    acc = accuracy_score(all_labels, binary_preds)
    cm = confusion_matrix(all_labels, binary_preds)

    return avg_loss, auc, acc,cm,all_labels, binary_preds

def plot_confusion_matrix(cm, save_path):
    """Saves a visual heatmap of the confusion matrix."""
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                     xticklabels=['REAL', 'FAKE'], yticklabels=['REAL', 'FAKE'])
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    plt.savefig(save_path)
    plt.close()

def main():
    print(f"Using device: {DEVICE}")

    metadata = pd.read_json(METADATA_PATH).T
    df = metadata.copy()
    df['label'] = df['label'].apply(lambda x: 1 if x == 'FAKE' else 0)
    
    labels = df['label'].tolist()
    num_real = labels.count(0)
    num_fake = labels.count(1)
    
    # pos_weight = torch.tensor([num_real / num_fake], device=DEVICE)
    
    print(f"Class Imbalance: {num_fake} FAKE / {num_real} REAL.")

    train_loader, val_loader= get_data_loaders()

    model = LRATF().to(DEVICE)
    
    # loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = FocalLoss(alpha=0.32, gamma=2.0)
    
    # <<< MODIFIED: Reduced weight_decay to a small, standard value >>>
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE,weight_decay=1e-5)
    
    # scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    best_val_auc = 0.0

    print("🚀 Starting Training...")
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")

        train_loss, train_auc, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, DEVICE)
        print(f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, Train Accuracy: {train_acc:.4f}")

        val_loss, val_auc, val_acc,val_cm,true_labels, pred_labels = validate_one_epoch(model, val_loader, loss_fn, DEVICE)
        print(f"Validation Loss: {val_loss:.4f}, Validation AUC: {val_auc:.4f}, Validation Accuracy: {val_acc:.4f}")
        
        # scheduler.step()
        # print(f"Current learning rate: {scheduler.get_last_lr()[0]}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            save_path = os.path.join(MODEL_SAVE_DIR, f'best_model_epoch_{epoch+1}_auc_{val_auc:.4f}.pth')
            torch.save(model.state_dict(), save_path)
            print(f"✅ Model saved to {save_path}")
            cm_save_path = os.path.join(MODEL_SAVE_DIR, f'best_cm_epoch_{epoch+1}.png')
            plot_confusion_matrix(val_cm, cm_save_path)
            print(f"✅ Confusion Matrix saved to {cm_save_path}")
        

        report = classification_report(true_labels, pred_labels, target_names=['REAL', 'FAKE'])
        print("Validation Classification Report:")
        print(report)


    

    print("\n--- ✅ Training Complete ---")
    print(f"Best validation AUC: {best_val_auc:.4f}")

if __name__ == "__main__":
    main()
