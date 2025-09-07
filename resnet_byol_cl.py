import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
import pywt
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
import json
import os
import copy
import torch.nn.functional as F

# ============ CONFIG ============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
val_fraction = 0.2
random_seed = 42
val_split_seed = 123

# BYOL Configuration
byol_epochs = 50
byol_lr = 0.003
byol_momentum = 0.996
projection_dim = 256
hidden_dim = 4096

# Fine-tuning Configuration  
finetune_epochs = 150
finetune_lr = 0.001
patience = 10
transition_epochs = 50  # Époques pour la transition wavelet → original
wavelet_name = 'bior3.3'

torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

print(f"🖥️  Device: {device}")
print(f"🔄 BYOL Pré-entraînement: {byol_epochs} époques")
print(f"🎓 Fine-tuning Curriculum: {finetune_epochs} époques")

# ============ TRANSFORMATIONS ============
# Transformations fortes pour BYOL (auto-supervisé)
byol_transform_1 = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

byol_transform_2 = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Transformations pour le fine-tuning
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# ============ WAVELET TRANSFORM ============
def wavelet_LL1(img_tensor, wavelet_name='bior3.3'):
    """
    Applique la transformée en ondelettes LL1 sur un tensor PyTorch
    """
    # Dénormaliser temporairement pour wavelet
    img_denorm = img_tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    img_denorm = torch.clamp(img_denorm, 0, 1)
    
    img_np = img_denorm.permute(1, 2, 0).numpy()
    channels_LL1 = []
    
    for c in range(3):
        cA1, _ = pywt.dwt2(img_np[:, :, c], wavelet=wavelet_name)
        # Redimensionner à la taille originale
        cA1_resized = np.array(Image.fromarray((cA1 * 255).astype(np.uint8)).resize((32, 32))) / 255.0
        channels_LL1.append(cA1_resized)
    
    merged = np.stack(channels_LL1, axis=2)
    img_tensor_new = torch.from_numpy(merged.transpose((2, 0, 1))).float()
    
    # Renormaliser
    img_tensor_new = (img_tensor_new - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    return img_tensor_new

# ============ ARCHITECTURE RESEAU ============
class CNNBackbone(nn.Module):
    """Backbone CNN pour BYOL (sans classification finale)"""
    def __init__(self):
        super(CNNBackbone, self).__init__()
        
        self.features = nn.Sequential(
            # Premier bloc
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Deuxième bloc
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Troisième bloc
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return x

class ClassificationModel(nn.Module):
    """Modèle complet avec backbone + tête de classification"""
    def __init__(self, backbone, num_classes=10):
        super(ClassificationModel, self).__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

# ============ BYOL COMPONENTS ============
class MLP(nn.Module):
    """MLP pour projections et prédictions BYOL"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class BYOL(nn.Module):
    """Bootstrap Your Own Latent (BYOL)"""
    def __init__(self, backbone, projection_dim=256, hidden_dim=4096, momentum=0.996):
        super(BYOL, self).__init__()
        
        self.momentum = momentum
        self.backbone_dim = 256  # Sortie du backbone CNN
        
        # Online network (celui qui apprend)
        self.online_encoder = backbone
        self.online_projector = MLP(self.backbone_dim, hidden_dim, projection_dim)
        self.online_predictor = MLP(projection_dim, hidden_dim, projection_dim)
        
        # Target network (EMA de l'online network)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        
        # Désactiver les gradients pour le target network
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False
    
    def update_target_network(self):
        """Mise à jour EMA du target network"""
        for online_params, target_params in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_params.data = self.momentum * target_params.data + (1 - self.momentum) * online_params.data
        
        for online_params, target_params in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            target_params.data = self.momentum * target_params.data + (1 - self.momentum) * online_params.data
    
    def forward(self, x1, x2):
        """Forward pass BYOL"""
        # Online network
        online_features_1 = self.online_encoder(x1)
        online_projections_1 = self.online_projector(online_features_1)
        online_predictions_1 = self.online_predictor(online_projections_1)
        
        online_features_2 = self.online_encoder(x2)
        online_projections_2 = self.online_projector(online_features_2)
        online_predictions_2 = self.online_predictor(online_projections_2)
        
        # Target network (pas de gradients)
        with torch.no_grad():
            target_features_1 = self.target_encoder(x1)
            target_projections_1 = self.target_projector(target_features_1)
            
            target_features_2 = self.target_encoder(x2)
            target_projections_2 = self.target_projector(target_features_2)
        
        return online_predictions_1, online_predictions_2, target_projections_1, target_projections_2

def byol_loss_fn(p1, p2, z1, z2):
    """Loss BYOL basée sur la similarité cosinus"""
    p1_norm = F.normalize(p1, dim=1)
    p2_norm = F.normalize(p2, dim=1)
    z1_norm = F.normalize(z1, dim=1)
    z2_norm = F.normalize(z2, dim=1)
    
    loss1 = 2 - 2 * (p1_norm * z2_norm).sum(dim=1).mean()
    loss2 = 2 - 2 * (p2_norm * z1_norm).sum(dim=1).mean()
    
    return (loss1 + loss2) / 2

# ============ DATASETS ============
class BYOLDataset(Dataset):
    """Dataset BYOL: retourne deux augmentations de la même image"""
    def __init__(self, base_dataset, transform1, transform2):
        self.base_dataset = base_dataset
        self.transform1 = transform1
        self.transform2 = transform2
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        if isinstance(self.base_dataset[idx], tuple):
            img, _ = self.base_dataset[idx]
        else:
            img = self.base_dataset[idx]
        
        # Deux augmentations différentes
        img1 = self.transform1(img)
        img2 = self.transform2(img)
        
        return img1, img2

class CurriculumDataset(Dataset):
    """Dataset avec curriculum wavelet → original"""
    def __init__(self, base_dataset, wavelet_ratio=1.0, wavelet_name='bior3.3'):
        self.base_dataset = base_dataset
        self.wavelet_ratio = wavelet_ratio
        self.wavelet_name = wavelet_name
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        
        # Appliquer wavelet selon le ratio
        if random.random() < self.wavelet_ratio:
            img = wavelet_LL1(img, self.wavelet_name)
        
        return img, label
    
    def update_wavelet_ratio(self, new_ratio):
        """Met à jour le ratio wavelet"""
        self.wavelet_ratio = new_ratio

# Chargement des données
print("📁 Chargement des données CIFAR-10...")

# Dataset pour BYOL (sans labels)
cifar10_unlabeled = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)

# Dataset pour fine-tuning (avec labels)
train_dataset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)

# Test set split
full_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
val_size = int(len(full_testset) * val_fraction)
test_size = len(full_testset) - val_size
val_set, final_test_set = random_split(full_testset, [val_size, test_size], 
                                      generator=torch.Generator().manual_seed(val_split_seed))

print(f"📊 BYOL: {len(cifar10_unlabeled)} échantillons")
print(f"📊 Train: {len(train_dataset)}, Val: {len(val_set)}, Test: {len(final_test_set)}")

# ============ PHASE 1: PRE-ENTRAINEMENT BYOL ============
def pretrain_byol():
    """Pré-entraînement BYOL auto-supervisé"""
    print(f"\n🔄 PHASE 1: Pré-entraînement BYOL ({byol_epochs} époques)")
    
    # Dataset et DataLoader BYOL
    byol_dataset = BYOLDataset(cifar10_unlabeled, byol_transform_1, byol_transform_2)
    byol_loader = DataLoader(byol_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Modèle BYOL
    backbone = CNNBackbone().to(device)
    byol_model = BYOL(backbone, projection_dim=projection_dim, 
                      hidden_dim=hidden_dim, momentum=byol_momentum).to(device)
    
    # Optimiseur pour les paramètres trainables seulement
    trainable_params = (list(byol_model.online_encoder.parameters()) + 
                       list(byol_model.online_projector.parameters()) + 
                       list(byol_model.online_predictor.parameters()))
    
    optimizer = optim.Adam(trainable_params, lr=byol_lr, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=byol_epochs)
    
    # Historique BYOL
    byol_history = {'loss': [], 'lr': []}
    
    # Boucle d'entraînement BYOL
    for epoch in range(1, byol_epochs + 1):
        byol_model.train()
        total_loss = 0
        num_batches = 0
        
        for x1, x2 in byol_loader:
            x1, x2 = x1.to(device), x2.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            p1, p2, z1, z2 = byol_model(x1, x2)
            
            # Loss BYOL
            loss = byol_loss_fn(p1, p2, z1, z2)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Mise à jour EMA target network
            byol_model.update_target_network()
            
            total_loss += loss.item()
            num_batches += 1
        
        scheduler.step()
        avg_loss = total_loss / num_batches
        
        byol_history['loss'].append(avg_loss)
        byol_history['lr'].append(optimizer.param_groups[0]['lr'])
        
        print(f"[BYOL] Epoch {epoch:3d}/{byol_epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Sauvegarder le backbone pré-entraîné
    torch.save(byol_model.online_encoder.state_dict(), 'byol_backbone.pth')
    print("💾 Backbone BYOL sauvegardé: byol_backbone.pth")
    
    return byol_model.online_encoder, byol_history

# ============ PHASE 2: FINE-TUNING AVEC CURRICULUM ============
def finetune_with_curriculum(pretrained_backbone):
    """Fine-tuning supervisé avec curriculum learning"""
    print(f"\n🎓 PHASE 2: Fine-tuning avec Curriculum ({finetune_epochs} époques)")
    
    # Créer le modèle de classification avec backbone pré-entraîné
    model = ClassificationModel(pretrained_backbone, num_classes=10).to(device)
    
    # Dataset curriculum
    curriculum_dataset = CurriculumDataset(train_dataset, wavelet_ratio=1.0, wavelet_name=wavelet_name)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Optimiseur et critère
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=finetune_lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=finetune_epochs)
    
    # Historique fine-tuning
    history = {
        'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
        'wavelet_ratio': [], 'lr': []
    }
    
    best_val_acc = 0
    patience_counter = 0
    
    # Boucle de fine-tuning
    for epoch in range(1, finetune_epochs + 1):
        # Calcul du ratio wavelet (transition graduelle)
        if epoch <= transition_epochs:
            wavelet_ratio = 1.0 - (epoch - 1) / transition_epochs  # 1.0 → 0.0
        else:
            wavelet_ratio = 0.0
        
        # Mise à jour du dataset
        curriculum_dataset.update_wavelet_ratio(wavelet_ratio)
        train_loader = DataLoader(curriculum_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        
        # === TRAINING ===
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # === VALIDATION ===
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        scheduler.step()
        
        # Enregistrer métriques
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['wavelet_ratio'].append(wavelet_ratio)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Early stopping et sauvegarde du meilleur modèle
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_byol_curriculum_model.pth')
        else:
            patience_counter += 1
        
        # Affichage des résultats
        phase = "Wavelet→Original" if wavelet_ratio > 0 else "Original"
        print(f"[{phase}] Epoch {epoch:3d}/{finetune_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"WR: {wavelet_ratio:.2f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"⏹️  Early stopping à l'époque {epoch}")
            break
    
    # Charger le meilleur modèle
    model.load_state_dict(torch.load('best_byol_curriculum_model.pth'))
    print(f"✅ Meilleure validation accuracy: {best_val_acc:.4f}")
    
    return model, history

# ============ EVALUATION FINALE ============
def evaluate_model(model):
    """Évaluation finale sur le test set"""
    print(f"\n🎯 ÉVALUATION FINALE")
    
    test_loader = DataLoader(final_test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    criterion = nn.CrossEntropyLoss()
    
    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)
    
    test_loss /= test_total
    test_acc = test_correct / test_total
    
    print(f"📊 Test Loss: {test_loss:.4f}")
    print(f"📊 Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    return test_acc, test_loss

# ============ VISUALISATION ============
def plot_results(byol_history, finetune_history):
    """Graphiques des résultats"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # BYOL Loss
    axes[0,0].plot(byol_history['loss'], color='purple', linewidth=2)
    axes[0,0].set_title('BYOL Pre-training Loss')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].grid(True, alpha=0.3)
    
    # BYOL Learning Rate
    axes[1,0].plot(byol_history['lr'], color='purple', linewidth=2)
    axes[1,0].set_title('BYOL Learning Rate')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('Learning Rate')
    axes[1,0].grid(True, alpha=0.3)
    
    # Fine-tuning Accuracy
    axes[0,1].plot(finetune_history['train_acc'], label='Train', color='blue', linewidth=2)
    axes[0,1].plot(finetune_history['val_acc'], label='Validation', color='red', linewidth=2)
    axes[0,1].set_title('Fine-tuning Accuracy')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Accuracy')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Fine-tuning Loss
    axes[1,1].plot(finetune_history['train_loss'], label='Train', color='blue', linewidth=2)
    axes[1,1].plot(finetune_history['val_loss'], label='Validation', color='red', linewidth=2)
    axes[1,1].set_title('Fine-tuning Loss')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('Loss')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # Wavelet Ratio Evolution
    axes[0,2].plot(finetune_history['wavelet_ratio'], color='green', linewidth=2)
    axes[0,2].set_title('Curriculum: Wavelet Ratio')
    axes[0,2].set_xlabel('Epoch')
    axes[0,2].set_ylabel('Wavelet Ratio')
    axes[0,2].grid(True, alpha=0.3)
    
    # Fine-tuning Learning Rate
    axes[1,2].plot(finetune_history['lr'], color='orange', linewidth=2)
    axes[1,2].set_title('Fine-tuning Learning Rate')
    axes[1,2].set_xlabel('Epoch')
    axes[1,2].set_ylabel('Learning Rate')
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('byol_curriculum_results.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============ EXECUTION PRINCIPALE ============
def main():
    print("🚀 BYOL + CURRICULUM LEARNING SUR CIFAR-10")
    print("="*60)
    
    # Phase 1: Pré-entraînement BYOL
    pretrained_backbone, byol_history = pretrain_byol()
    
    # Phase 2: Fine-tuning avec curriculum
    final_model, finetune_history = finetune_with_curriculum(pretrained_backbone)
    
    # Phase 3: Évaluation finale
    test_acc, test_loss = evaluate_model(final_model)
    
    # Résultats finaux
    print("\n" + "="*60)
    print("📊 RÉSULTATS FINAUX")
    print("="*60)
    print(f"🎯 Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"🎯 Test Loss: {test_loss:.4f}")
    
    # Sauvegarde des résultats
    results = {
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'byol_history': byol_history,
        'finetune_history': finetune_history,
        'config': {
            'byol_epochs': byol_epochs,
            'finetune_epochs': finetune_epochs,
            'transition_epochs': transition_epochs,
            'wavelet_name': wavelet_name,
            'byol_lr': byol_lr,
            'finetune_lr': finetune_lr
        }
    }
    
    with open('byol_curriculum_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Graphiques
    plot_results(byol_history, finetune_history)
    
    print("💾 Résultats sauvegardés dans byol_curriculum_results.json")
    print("📈 Graphiques sauvegardés dans byol_curriculum_results.png")
    
    return results

if __name__ == "__main__":
    results = main()
