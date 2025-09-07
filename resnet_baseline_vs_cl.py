import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset
import pywt
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import os

# ============ CONFIG ============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
learning_rate = 0.001
num_epochs_total = 150
patience = 10
use_extended_dataset = False  # 🔁 Change ici pour utiliser le dataset étendu ou non
val_fraction = 0.2
random_seed = 42
val_split_seed = 123

# Curriculum Learning params
use_curriculum = True  # Pour comparer avec baseline
transition_epochs = 50  # Nombre d'époques pour la transition graduelle
wavelet_name = 'bior3.3'  # Peut tester 'db4', 'haar', etc.

torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

print(f"🖥️  Device: {device}")
print(f"📊 Dataset étendu: {'Oui' if use_extended_dataset else 'Non'}")
print(f"📚 Curriculum Learning: {'Activé' if use_curriculum else 'Désactivé'}")

# ============ TRANSFORMATIONS ============
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet stats
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# ============ WAVELET TRANSFORM ============
def wavelet_LL1(img_tensor, wavelet_name='bior3.3'):
    """
    Applique la transformée en ondelettes sur un tensor PyTorch
    Args:
        img_tensor: Tensor de shape (C, H, W)
        wavelet_name: Type d'ondelette à utiliser
    Returns:
        Tensor transformé de même shape
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

# ============ DATASET AVEC CURRICULUM ============
class CurriculumDataset(Dataset):
    def __init__(self, base_dataset, wavelet_ratio=0.0, wavelet_name='bior3.3'):
        """
        Dataset qui mélange images originales et transformées par ondelettes
        Args:
            base_dataset: Dataset de base (CIFAR-10)
            wavelet_ratio: Proportion d'images wavelet (0.0 = toutes originales, 1.0 = toutes wavelet)
            wavelet_name: Type d'ondelette
        """
        self.base_dataset = base_dataset
        self.wavelet_ratio = wavelet_ratio
        self.wavelet_name = wavelet_name
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        
        # Décider si on applique la transformée wavelet
        if random.random() < self.wavelet_ratio:
            img = wavelet_LL1(img, self.wavelet_name)
        
        return img, label
    
    def update_wavelet_ratio(self, new_ratio):
        """Met à jour le ratio d'images wavelet"""
        self.wavelet_ratio = new_ratio

# ============ MODÈLE AMÉLIORÉ ============
class ImprovedCNN(nn.Module):
    """CNN adapté pour CIFAR-10 avec Batch Normalization"""
    def __init__(self, num_classes=10):
        super(ImprovedCNN, self).__init__()
        
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
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def get_model(model_type='improved_cnn'):
    """Retourne le modèle choisi"""
    if model_type == 'resnet18':
        model = torchvision.models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, 10)
    else:  # improved_cnn
        model = ImprovedCNN(num_classes=10)
    
    return model.to(device)

# ============ DATASETS ============
print("📁 Chargement des données...")

if use_extended_dataset:
    try:
        saved = torch.load('./data/cifar10_extended.pt')
        data = saved['data']
        labels = saved['labels']
        print(f"✅ Dataset étendu chargé: {len(data)} échantillons")
        
        # Appliquer les transformations manuellement si nécessaire
        raw_train_data = [(data[i], labels[i].item()) for i in range(len(data))]
        train_dataset_base = raw_train_data
    except FileNotFoundError:
        print("⚠️  Fichier cifar10_extended.pt non trouvé, utilisation du dataset standard")
        use_extended_dataset = False

if not use_extended_dataset:
    train_dataset_raw = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_dataset_base = train_dataset_raw
    print(f"✅ Dataset CIFAR-10 standard chargé: {len(train_dataset_base)} échantillons")

# Test set split
full_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
val_size = int(len(full_testset) * val_fraction)
test_size = len(full_testset) - val_size
val_set, final_test_set = random_split(full_testset, [val_size, test_size], 
                                      generator=torch.Generator().manual_seed(val_split_seed))

print(f"📊 Train: {len(train_dataset_base)}, Val: {len(val_set)}, Test: {len(final_test_set)}")

# ============ TRAINING & EVAL FUNCTIONS ============
def train_epoch(model, dataloader, criterion, optimizer, scheduler=None):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    if scheduler:
        scheduler.step()
        
    return total_loss / total, correct / total

def eval_epoch(model, dataloader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    return total_loss / total, correct / total

# ============ CURRICULUM TRAINING ============
def train_curriculum_improved():
    """Entraînement avec curriculum learning amélioré"""
    print("\n🚀 Début du Curriculum Learning Amélioré")
    
    # Créer le dataset avec curriculum
    curriculum_dataset = CurriculumDataset(train_dataset_base, wavelet_ratio=1.0, wavelet_name=wavelet_name)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    # Modèle et optimiseur
    model = get_model('improved_cnn')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs_total)
    
    # Métriques pour tracking
    history = {
        'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 
        'wavelet_ratio': [], 'lr': []
    }
    
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(1, num_epochs_total + 1):
        # Calcul du ratio wavelet (transition graduelle)
        if epoch <= transition_epochs:
            wavelet_ratio = 1.0 - (epoch - 1) / transition_epochs  # 1.0 -> 0.0
        else:
            wavelet_ratio = 0.0
            
        # Mise à jour du dataset
        curriculum_dataset.update_wavelet_ratio(wavelet_ratio)
        train_loader = DataLoader(curriculum_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        
        # Entraînement
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion)
        
        # Enregistrer métriques
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['wavelet_ratio'].append(wavelet_ratio)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_curriculum_model.pth')
        else:
            patience_counter += 1
        
        # Affichage
        phase = "LL1→Original" if wavelet_ratio > 0 else "Original"
        print(f"[{phase}] Epoch {epoch:3d} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"WR: {wavelet_ratio:.2f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if patience_counter >= patience:
            print(f"⏹️  Early stopping à l'époque {epoch}")
            break
    
    # Charger le meilleur modèle
    model.load_state_dict(torch.load('best_curriculum_model.pth'))
    return model, history

def train_baseline():
    """Entraînement baseline sans curriculum learning"""
    print("\n📊 Entraînement Baseline (sans curriculum)")
    
    train_loader = DataLoader(train_dataset_base, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    model = get_model('improved_cnn')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs_total)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(1, num_epochs_total + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_baseline_model.pth')
        else:
            patience_counter += 1
            
        print(f"[Baseline] Epoch {epoch:3d} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
              
        if patience_counter >= patience:
            print(f"⏹️  Early stopping à l'époque {epoch}")
            break
    
    model.load_state_dict(torch.load('best_baseline_model.pth'))
    return model, history

# ============ EVALUATION FINALE ============
def test_model(model, model_name="Model"):
    print(f"\n🎯 Évaluation finale - {model_name}")
    test_loader = DataLoader(final_test_set, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = eval_epoch(model, test_loader, criterion)
    print(f"📊 {model_name} - Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
    return test_acc

def plot_comparison(curriculum_history, baseline_history):
    """Graphiques de comparaison"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0,0].plot(curriculum_history['val_acc'], label='Curriculum', color='blue')
    axes[0,0].plot(baseline_history['val_acc'], label='Baseline', color='red')
    axes[0,0].set_title('Validation Accuracy')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # Loss
    axes[0,1].plot(curriculum_history['val_loss'], label='Curriculum', color='blue')
    axes[0,1].plot(baseline_history['val_loss'], label='Baseline', color='red')
    axes[0,1].set_title('Validation Loss')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # Wavelet ratio
    axes[1,0].plot(curriculum_history['wavelet_ratio'], color='green')
    axes[1,0].set_title('Wavelet Ratio (Curriculum)')
    axes[1,0].set_ylabel('Ratio')
    axes[1,0].grid(True)
    
    # Learning rates
    axes[1,1].plot(curriculum_history['lr'], label='Curriculum', color='blue')
    axes[1,1].plot(baseline_history['lr'], label='Baseline', color='red')
    axes[1,1].set_title('Learning Rate')
    axes[1,1].legend()
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.savefig('curriculum_vs_baseline.png', dpi=300)
    plt.show()

# ============ EXECUTION PRINCIPALE ============
def main():
    results = {}
    
    if use_curriculum:
        # Entraînement avec curriculum
        curriculum_model, curriculum_history = train_curriculum_improved()
        curriculum_test_acc = test_model(curriculum_model, "Curriculum")
        results['curriculum'] = {
            'test_accuracy': curriculum_test_acc,
            'history': curriculum_history
        }
    
    # Entraînement baseline
    baseline_model, baseline_history = train_baseline()
    baseline_test_acc = test_model(baseline_model, "Baseline")
    results['baseline'] = {
        'test_accuracy': baseline_test_acc,
        'history': baseline_history
    }
    
    # Comparaison
    print("\n" + "="*50)
    print("📊 RÉSULTATS FINAUX")
    print("="*50)
    if use_curriculum:
        print(f"🎓 Curriculum Learning: {curriculum_test_acc:.4f}")
    print(f"📊 Baseline:           {baseline_test_acc:.4f}")
    
    if use_curriculum:
        improvement = curriculum_test_acc - baseline_test_acc
        print(f"📈 Amélioration:       {improvement:+.4f}")
        
        # Graphiques
        plot_comparison(curriculum_history, baseline_history)
    
    # Sauvegarder résultats
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("💾 Résultats sauvegardés dans results.json")
    return results

# ============ RUN ============
if __name__ == "__main__":
    results = main()
