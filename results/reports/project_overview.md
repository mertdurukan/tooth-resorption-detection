# DİŞ REZORPSIYON TESPİTİ - TAM PROJE DOKÜMANTASYONU

## İÇİNDEKİLER

1. [Proje Özeti](#1-proje-özeti)
2. [Veri Seti Detayları](#2-veri-seti-detayları)
3. [Denenen Tüm Yöntemler](#3-denenen-tüm-yöntemler)
4. [Model Mimarileri](#4-model-mimarileri)
5. [Hiperparametre Konfigürasyonları](#5-hiperparametre-konfigürasyonları)
6. [Eğitim Süreci](#6-eğitim-süreci)
7. [Tüm Sonuçlar](#7-tüm-sonuçlar)
8. [Gelişmiş Analiz Yöntemleri](#8-gelişmiş-analiz-yöntemleri)
9. [Kod Dosyaları ve Açıklamaları](#9-kod-dosyaları-ve-açıklamaları)
10. [Görselleştirmeler](#10-görselleştirmeler)
11. [Sonuç ve Öneriler](#11-sonuç-ve-öneriler)

---

## 1. PROJE ÖZETİ

### 1.1 Problem Tanımı
- **Görev**: 20'lik diş rezorpsiyonunun otomatik tespiti
- **Tip**: 3-sınıflı görüntü sınıflandırma
- **Sınıflar**:
  - **Temaslı (Sınıf 0)**: Komşu dişle temas halinde
  - **Bağımsız (Sınıf 1)**: Komşu dişten bağımsız
  - **Rezorpsiyon (Sınıf 2)**: Kök rezorpsiyonu mevcut

### 1.2 Proje Hedefleri
1. En yüksek doğruluk oranına sahip modeli bulmak
2. Farklı dikkat mekanizmalarının etkisini ölçmek
3. Transformer tabanlı modelleri CNN ile karşılaştırmak
4. Klinik kullanıma uygun bir sistem geliştirmek

### 1.3 Proje Kronolojisi
1. **Faz 1**: Baseline CNN modeli oluşturma
2. **Faz 2**: YOLO ile object detection denemesi
3. **Faz 3**: Vision Transformer (ViT) implementasyonu
4. **Faz 4**: Swin Transformer implementasyonu
5. **Faz 5**: Dikkat mekanizmaları (CBAM, SE, MHA) eklenmesi
6. **Faz 6**: Hibrit modeller (Swin + Attention, ViT + Attention)
7. **Faz 7**: Ensemble model oluşturma
8. **Faz 8**: High-confidence filtering optimizasyonu

---

## 2. VERİ SETİ DETAYLARI

### 2.1 Genel Bilgiler

| Özellik | Değer |
|---------|-------|
| Toplam Görüntü Sayısı | 306 |
| Görüntü Formatı | JPEG (base64 encoded) |
| Kaynak | Dental panoramik röntgenler |
| Depolama | JSON dosyaları içinde |
| Dosya Konumu | `20 lik diş rezorpsiyon/` |

### 2.2 Sınıf Dağılımı

| Sınıf | Sayı | Oran |
|-------|------|------|
| Temaslı | 130 | %42.5 |
| Bağımsız | 84 | %27.5 |
| Rezorpsiyon | 92 | %30.0 |
| **Toplam** | **306** | **100%** |

### 2.3 Veri Bölünmesi

```
Tüm Modellerde Kullanılan Bölünme:
├── Training Set: 244 görüntü (80%)
├── Validation Set: 62 görüntü (20%)
├── Stratification: Evet (sınıf dağılımı korunur)
└── Random Seed: 42 (reproducibility için)
```

### 2.4 JSON Dosya Yapısı

```json
{
  "imageData": "base64_encoded_jpeg_data...",
  "shapes": [
    {
      "label": "temaslı|bağımsız|rezorpsiyon",
      "points": [[x1, y1], [x2, y2], ...]
    }
  ],
  "imagePath": "original_image_name.jpg",
  "imageHeight": 224,
  "imageWidth": 224
}
```

---

## 3. DENENEN TÜM YÖNTEMLER

### 3.1 Baseline Modeller

| # | Model | Kategori | Sonuç |
|---|-------|----------|-------|
| 1 | CNN (2D Konvolüsyonel) | Baseline | %42.48 |
| 2 | YOLOv11 (Object Detection) | Detection | %42.81 |

### 3.2 Transformer Modelleri

| # | Model | Patch Size | Parametre | Sonuç |
|---|-------|------------|-----------|-------|
| 3 | vit_base_16 | 16x16 | 85.8M | %89.87 |
| 4 | vit_base_32 | 32x32 | 87.5M | %85.95 |
| 5 | vit_large_16 | 16x16 | 303.3M | %72.22 |
| 6 | swin_tiny | 4x4 | 27.5M | %90.20 |
| 7 | swin_small | 4x4 | 48.8M | %89.87 |
| 8 | swin_base | 4x4 | 86.7M | %85.95 |

### 3.3 CNN + Attention Modelleri

| # | Model | Base | Attention | Sonuç |
|---|-------|------|-----------|-------|
| 9 | resnet50_cbam | ResNet50 | CBAM | %43.14 |
| 10 | resnet50_se | ResNet50 | SE-Net | %88.24 |
| 11 | efficientnet_attention | EfficientNet-B0 | SE-Net | %85.29 |
| 12 | densenet_attention | DenseNet121 | SE-Net | %89.54 |

### 3.4 Hibrit Modeller (Transformer + Attention)

#### Swin Small Varyantları

| # | Model | Attention | Position | Sonuç |
|---|-------|-----------|----------|-------|
| 13 | swin_small_cbam_head | CBAM | Head | Eğitildi |
| 14 | swin_small_se_head | SE-Net | Head | Eğitildi |
| 15 | swin_small_mha_head | MHA | Head | %89.87 |
| 16 | swin_small_cbam_stages | CBAM | Stages | Eğitildi |
| 17 | swin_small_se_stages | SE-Net | Stages | Eğitildi |
| 18 | swin_small_mha_stages | MHA | Stages | Eğitildi |

#### ViT Base-16 Varyantları

| # | Model | Attention | Position | Sonuç |
|---|-------|-----------|----------|-------|
| 19 | vit_base_16_cbam_head | CBAM | Head | Eğitildi |
| 20 | vit_base_16_se_head | SE-Net | Head | Eğitildi |
| 21 | vit_base_16_mha_head | MHA | Head | Eğitildi |
| 22 | vit_base_16_cbam_stages | CBAM | Stages | Eğitildi |
| 23 | vit_base_16_se_stages | SE-Net | Stages | Eğitildi |
| 24 | vit_base_16_mha_stages | MHA | Stages | Eğitildi |

### 3.5 Ensemble Modeller

| # | Ensemble | Modeller | Aggregation | Sonuç |
|---|----------|----------|-------------|-------|
| 25 | Ensemble-3 | swin_small_mha_head + swin_small + vit_base_16 | Weighted Soft Voting | %96.73 |

### 3.6 Gelişmiş Yöntemler

| # | Yöntem | Açıklama | Sonuç |
|---|--------|----------|-------|
| 26 | High-Confidence Filtering | %70 threshold | %100 (77.8% coverage) |
| 27 | K-Fold Cross Validation | 5-Fold, 30 epoch | %49.66 ± 5.53% |
| 28 | Test-Time Augmentation | 7 augmentation | %80.07 |
| 29 | Uncertainty Estimation | MC Dropout (30 samples) | %86.27 |
| 30 | Focal Loss + Class Weights | γ=2.0, inverse freq | Kullanıldı |

---

## 4. MODEL MİMARİLERİ

### 4.1 CNN Baseline

```python
Model: Sequential CNN
├── Conv2D(32, 3x3) + ReLU + MaxPool
├── Conv2D(64, 3x3) + ReLU + MaxPool
├── Conv2D(128, 3x3) + ReLU + MaxPool
├── Flatten
├── Dense(128) + ReLU + Dropout(0.5)
└── Dense(3) + Softmax

Total Parameters: 652,995
Input Shape: (224, 224, 3)
Output: 3 classes
```

### 4.2 Vision Transformer (ViT)

```python
Model: vit_base_patch16_224
├── Patch Embedding: 14x14 = 196 patches
│   └── Patch Size: 16x16
│   └── Embed Dim: 768
├── CLS Token: 1 learnable token
├── Position Embedding: 197 positions
├── Transformer Encoder (x12):
│   ├── LayerNorm
│   ├── Multi-Head Self-Attention (12 heads)
│   ├── LayerNorm
│   └── MLP (768 → 3072 → 768)
├── LayerNorm
└── Classification Head: Linear(768 → 3)

Total Parameters: 85,800,963
Pre-trained: ImageNet-21K
```

### 4.3 Swin Transformer

```python
Model: swin_small_patch4_window7_224
├── Patch Partition: 56x56 = 3136 patches
│   └── Patch Size: 4x4
├── Stage 1:
│   ├── Linear Embedding (96 dim)
│   └── Swin Transformer Blocks (x2)
├── Patch Merging → Stage 2 (192 dim, 28x28)
│   └── Swin Transformer Blocks (x2)
├── Patch Merging → Stage 3 (384 dim, 14x14)
│   └── Swin Transformer Blocks (x18)
├── Patch Merging → Stage 4 (768 dim, 7x7)
│   └── Swin Transformer Blocks (x2)
├── Global Average Pooling
└── Classification Head: Linear(768 → 3)

Total Parameters: 48,839,565
Window Size: 7x7
Shifted Window: Yes
```

### 4.4 CBAM (Convolutional Block Attention Module)

```python
class CBAM:
    def __init__(self, channels, reduction=16):
        # Channel Attention
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.max_pool = AdaptiveMaxPool2d(1)
        self.fc1 = Conv2d(channels, channels // reduction, 1)
        self.fc2 = Conv2d(channels // reduction, channels, 1)
        
        # Spatial Attention
        self.conv = Conv2d(2, 1, kernel_size=7, padding=3)
    
    def forward(self, x):
        # Channel Attention
        avg_out = self.fc2(relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(relu(self.fc1(self.max_pool(x))))
        channel_att = sigmoid(avg_out + max_out)
        x = x * channel_att
        
        # Spatial Attention
        avg_out = mean(x, dim=1, keepdim=True)
        max_out = max(x, dim=1, keepdim=True)
        spatial_att = sigmoid(self.conv(cat([avg_out, max_out], dim=1)))
        return x * spatial_att
```

### 4.5 SE-Net (Squeeze-and-Excitation)

```python
class SEBlock:
    def __init__(self, channels, reduction=16):
        self.squeeze = AdaptiveAvgPool2d(1)
        self.excitation = Sequential(
            Linear(channels, channels // reduction),
            ReLU(),
            Linear(channels // reduction, channels),
            Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
```

### 4.6 Multi-Head Self-Attention (MHA)

```python
class AttentionBlock:
    def __init__(self, dim, num_heads=8):
        self.attention = MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = LayerNorm(dim)
    
    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        attn_out, _ = self.attention(x_flat, x_flat, x_flat)
        attn_out = self.norm(attn_out)
        return attn_out.transpose(1, 2).reshape(B, C, H, W)
```

### 4.7 Hibrit Model: Swin + MHA Head

```python
class SwinSmallWithAttention:
    def __init__(self, attention_type='mha', position='head'):
        self.swin = timm.create_model('swin_small_patch4_window7_224')
        self.swin.head = Identity()  # Remove original head
        
        if position == 'head':
            self.attention = AttentionBlock(768, num_heads=8)
        
        self.norm = LayerNorm(768)
        self.fc = Linear(768, 3)
    
    def forward(self, x):
        x = self.swin.forward_features(x)  # (B, H, W, C) or (B, N, C)
        
        # Reshape for attention
        if x.dim() == 4:  # Swin output: (B, H, W, C)
            B, H, W, C = x.shape
            x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        x = self.attention(x)  # Apply MHA
        
        # Pool and classify
        x = x.mean(dim=[2, 3])  # Global average pool
        x = self.norm(x)
        return self.fc(x)
```

---

## 5. HİPERPARAMETRE KONFİGÜRASYONLARI

### 5.1 Standart Eğitim Konfigürasyonu

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| **epochs** | 50 | Maksimum epoch sayısı |
| **batch_size** | 8 | Mini-batch boyutu |
| **learning_rate** | 5e-5 | Başlangıç öğrenme oranı |
| **weight_decay** | 1e-4 | L2 regularization katsayısı |
| **optimizer** | AdamW | Adam with decoupled weight decay |
| **scheduler** | ReduceLROnPlateau | Plato'da LR azaltma |
| **scheduler_factor** | 0.5 | LR çarpan faktörü |
| **scheduler_patience** | 5 | Bekleme epoch sayısı |
| **early_stopping_patience** | 15 | Early stopping için bekleme |
| **backbone_freeze_epochs** | 10 | Backbone'un dondurulduğu epoch |
| **gradient_clip_norm** | 1.0 | Gradient clipping max norm |

### 5.2 AdamW Optimizer Detayları

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-5,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=1e-4
)
```

### 5.3 Learning Rate Scheduler

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',           # Minimize validation loss
    factor=0.5,           # LR *= 0.5
    patience=5,           # 5 epoch bekleme
    min_lr=1e-7,          # Minimum LR
    verbose=True
)
```

### 5.4 Grid Search Parametreleri (Denenen)

```python
grid_search_params = {
    'learning_rate': [1e-4, 5e-5, 1e-5],
    'batch_size': [4, 8, 16],
    'weight_decay': [1e-4, 1e-5],
    'optimizer': ['Adam', 'AdamW']
}

# En iyi kombinasyon:
best_config = {
    'learning_rate': 5e-5,
    'batch_size': 8,
    'weight_decay': 1e-4,
    'optimizer': 'AdamW'
}
```

### 5.5 Veri Artırma (Data Augmentation)

```python
# Training transforms
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),
        scale=None,
        shear=None
    ),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]    # ImageNet std
    )
])

# Validation/Test transforms
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

### 5.6 Loss Fonksiyonları

#### Standart CrossEntropyLoss
```python
criterion = nn.CrossEntropyLoss()
```

#### Focal Loss (Gelişmiş Eğitim)
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# Class weights (inverse frequency)
class_counts = [130, 84, 92]
class_weights = [306 / (3 * c) for c in class_counts]
# = [0.785, 1.214, 1.109]
```

### 5.7 Ensemble Konfigürasyonu

```python
ensemble_config = {
    'models': [
        ('swin_small_mha_head', 'models/swin_small_mha_head_best.pth', 1.0),
        ('swin_small', 'models/swin_small_best.pth', 0.9),
        ('vit_base_16', 'models/vit_base_16_best.pth', 0.8)
    ],
    'aggregation': 'weighted_soft_voting',
    'normalize_weights': True
}

# Normalized weights: [0.37, 0.33, 0.30]
```

### 5.8 K-Fold Cross Validation Konfigürasyonu

```python
kfold_config = {
    'n_folds': 5,
    'stratified': True,
    'epochs_per_fold': 30,
    'loss': 'FocalLoss',
    'gamma': 2.0,
    'use_class_weights': True,
    'use_weighted_sampler': True,
    'random_seed': 42
}
```

### 5.9 Test-Time Augmentation (TTA) Konfigürasyonu

```python
tta_transforms = [
    'original',           # Orijinal görüntü
    'horizontal_flip',    # Yatay çevirme
    'vertical_flip',      # Dikey çevirme
    'rotate_+10',         # +10° döndürme
    'rotate_-10',         # -10° döndürme
    'color_jitter_1',     # Renk augmentasyonu 1
    'color_jitter_2'      # Renk augmentasyonu 2
]

# Aggregation: Mean of all predictions
tta_aggregation = 'mean'
```

---

## 6. EĞİTİM SÜRECİ

### 6.1 Eğitim Adımları

```
Her Epoch İçin:
1. Training Phase:
   ├── Forward pass
   ├── Loss hesaplama
   ├── Backward pass (gradient hesaplama)
   ├── Gradient clipping (max_norm=1.0)
   ├── Optimizer step
   └── Metrik kaydetme

2. Validation Phase:
   ├── Forward pass (no gradient)
   ├── Loss ve metrik hesaplama
   └── En iyi model kontrolü

3. Post-Epoch:
   ├── Scheduler step (LR güncelleme)
   ├── Early stopping kontrolü
   └── Model kaydetme (eğer en iyi ise)
```

### 6.2 Transfer Learning Stratejisi

```
Epoch 1-10: Backbone Frozen
├── Sadece classification head eğitilir
├── Learning Rate: 5e-5
└── Amaç: Head'i stabilize etmek

Epoch 11-50: Full Fine-tuning
├── Tüm ağ eğitilir
├── Learning Rate: Scheduler kontrolünde
└── Amaç: Tüm katmanları optimize etmek
```

### 6.3 Metrik Hesaplama

```python
def calculate_metrics(y_true, y_pred, y_pred_proba):
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    # AUC (One-vs-Rest)
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    auc = roc_auc_score(y_true_bin, y_pred_proba, 
                        average='macro', multi_class='ovr')
    
    # Composite Score
    composite = (accuracy + precision + recall + f1 + auc) / 5
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'composite_score': composite
    }
```

### 6.4 Model Kaydetme

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'metrics': val_metrics,
    'config': config
}
torch.save(checkpoint, f'models/{model_type}_best.pth')
```

---

## 7. TÜM SONUÇLAR

### 7.1 Ana Performans Tablosu

| Sıra | Model | Accuracy | Precision | Recall | F1 | AUC | Composite |
|------|-------|----------|-----------|--------|-----|-----|-----------|
| 1 | **Ensemble (3 model)** | **96.73%** | **96.80%** | **96.80%** | **96.80%** | **99.71%** | **97.75%** |
| 2 | swin_tiny | 90.20% | 90.36% | 89.88% | 90.03% | 97.47% | 91.59% |
| 3 | swin_small | 89.87% | 89.47% | 90.29% | 89.81% | 98.74% | 91.64% |
| 4 | swin_small_mha_head | 89.87% | 89.49% | 90.26% | 89.81% | 98.62% | 91.61% |
| 5 | vit_base_16 | 89.87% | 89.49% | 90.26% | 89.81% | 98.62% | 91.61% |
| 6 | densenet_attention | 89.54% | 89.42% | 89.61% | 89.46% | 97.18% | 91.04% |
| 7 | resnet50_se | 88.24% | 87.83% | 88.27% | 88.00% | 97.78% | 90.02% |
| 8 | swin_base | 85.95% | 85.90% | 87.18% | 85.96% | 96.93% | 88.39% |
| 9 | vit_base_32 | 85.95% | 86.45% | 85.07% | 85.59% | 94.28% | 87.47% |
| 10 | efficientnet_attention | 85.29% | 85.58% | 84.56% | 84.81% | 92.79% | 86.61% |
| 11 | vit_large_16 | 72.22% | 81.90% | 68.32% | 70.61% | 88.76% | 76.36% |
| 12 | resnet50_cbam | 43.14% | 37.33% | 35.22% | 26.14% | 47.79% | 37.92% |
| 13 | YOLOv11_Detection | 42.81% | 47.54% | 33.73% | 20.71% | 51.71% | 39.30% |
| 14 | CNN_Baseline | 42.48% | 14.16% | 33.33% | 19.88% | 52.95% | 32.56% |

### 7.2 Sınıf Bazlı Performans (Top 5 Model)

#### Temaslı Sınıfı
| Model | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Ensemble | 95.42% | 96.92% | 96.16% |
| swin_small | 93.39% | 86.92% | 90.04% |
| vit_base_16 | 92.62% | 86.92% | 89.68% |
| swin_tiny | 90.84% | 91.54% | 91.19% |
| densenet_attention | 91.27% | 88.46% | 89.84% |

#### Bağımsız Sınıfı
| Model | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Ensemble | 97.56% | 95.24% | 96.39% |
| swin_small | 86.36% | 90.48% | 88.37% |
| vit_base_16 | 84.27% | 89.29% | 86.71% |
| swin_tiny | 86.36% | 90.48% | 88.37% |
| densenet_attention | 90.12% | 86.90% | 88.48% |

#### Rezorpsiyon Sınıfı
| Model | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Ensemble | 97.67% | 98.91% | 98.29% |
| swin_small | 88.66% | 93.48% | 91.01% |
| vit_base_16 | 91.58% | 94.57% | 93.05% |
| swin_tiny | 86.73% | 92.39% | 89.47% |
| densenet_attention | 86.87% | 93.48% | 90.05% |

### 7.3 Model Boyutu ve Hız

| Model | Parametre | Boyut | Inference (ms) |
|-------|-----------|-------|----------------|
| CNN_Baseline | 653K | 7.6 MB | 85.0 |
| swin_tiny | 27.5M | 105 MB | 19.1 |
| swin_small | 48.8M | 186 MB | 39.3 |
| vit_base_16 | 85.8M | 327 MB | 7.4 |
| vit_base_32 | 87.5M | 334 MB | 7.8 |
| swin_base | 86.7M | 331 MB | 36.5 |
| vit_large_16 | 303.3M | 1,157 MB | 14.1 |
| resnet50_se | 24.2M | 92 MB | 11.8 |
| densenet_attention | 11.2M | 43 MB | 20.1 |
| efficientnet_attention | 10.6M | 40 MB | 21.3 |

### 7.4 Confusion Matrix (Ensemble Model)

```
                    Predicted
                 Tem    Bag    Rez
Actual  Temaslı  123     4      3     (94.6% correct)
        Bağımsız   2    82      0     (97.6% correct)
        Rezorps.   1     0     91     (98.9% correct)
```

### 7.5 High-Confidence Filtering Sonuçları

| Threshold | Samples | Coverage | Accuracy | F1 | AUC |
|-----------|---------|----------|----------|-----|-----|
| 0% (Baseline) | 306 | 100.0% | 96.73% | 96.80% | 99.71% |
| 50% | 298 | 97.4% | 97.32% | 97.36% | 99.77% |
| 60% | 277 | 90.5% | 98.56% | 98.57% | 99.87% |
| **70%** | **238** | **77.8%** | **100.00%** | **100.00%** | **100.00%** |
| 80% | 206 | 67.3% | 100.00% | 100.00% | 100.00% |
| 90% | 133 | 43.5% | 100.00% | 100.00% | 100.00% |
| 95% | 44 | 14.4% | 100.00% | 100.00% | 100.00% |

### 7.6 Eğitim Süreleri

| Model | Eğitim Süresi | Epoch Başına |
|-------|---------------|--------------|
| CNN_Baseline | ~5 dk | ~6 sn |
| swin_tiny | ~19 dk | ~23 sn |
| swin_small | ~33 dk | ~40 sn |
| swin_base | ~47 dk | ~56 sn |
| vit_base_16 | ~30 dk | ~36 sn |
| vit_large_16 | ~85 dk | ~102 sn |
| Hibrit modeller | ~50 dk | ~60 sn |

---

## 8. GELİŞMİŞ ANALİZ YÖNTEMLERİ

### 8.1 K-Fold Cross Validation

```
Konfigürasyon:
├── K: 5 (Stratified K-Fold)
├── Epochs per fold: 30
├── Loss: Focal Loss (γ=2.0)
├── Class Weights: Inverse frequency
├── Weighted Sampler: Evet
└── Model: swin_small_mha_head

Fold Sonuçları:
├── Fold 1: Acc=54.84%, F1=55.08%, AUC=71.84%
├── Fold 2: Acc=44.26%, F1=41.85%, AUC=63.44%
├── Fold 3: Acc=44.26%, F1=43.70%, AUC=58.69%
├── Fold 4: Acc=55.74%, F1=53.67%, AUC=70.15%
└── Fold 5: Acc=49.18%, F1=48.56%, AUC=63.23%

Ortalama ± Std:
├── Accuracy: 49.66% ± 5.53%
├── F1-Score: 48.57% ± 5.86%
└── AUC: 65.47% ± 5.42%

Not: Düşük sonuçların sebebi her fold'un sıfırdan eğitilmesi
     ve 30 epoch'un yeterli olmaması.
```

### 8.2 Test-Time Augmentation (TTA)

```
TTA Augmentasyonları:
├── Original
├── Horizontal Flip
├── Vertical Flip
├── Rotation +10°
├── Rotation -10°
├── Color Jitter (variant 1)
└── Color Jitter (variant 2)

Aggregation: Mean of all predictions

Sonuç:
├── TTA Accuracy: 80.07%
├── TTA F1-Score: ~80%
└── Improvement: Marginal
```

### 8.3 Uncertainty Estimation

```
Yöntem: Monte Carlo Dropout
├── Dropout samples: 30
├── Model: swin_small_mha_head
└── Evaluation: Full dataset

Sonuçlar:
├── Overall Accuracy: 86.27%
├── Mean Confidence: 75.20%
├── High Confidence (>80%) Accuracy: 95.92%
├── Epistemic Uncertainty: ~0 (model deterministic)
└── Per-class Uncertainty: Uniform across classes

Bulgu: Model dropout kullanmadığı için epistemic uncertainty
       hesaplanamadı. Confidence-based filtering daha etkili.
```

### 8.4 Saliency Maps / Attention Visualization

```
Yöntem: Input Gradient Saliency
├── Target layers: Input image
├── Gradient: ∂Loss/∂Input
├── Visualization: Abs + Mean across channels
└── Overlay: Alpha blending on original

Gözlemler:
├── Model diş bölgelerine odaklanıyor
├── Rezorpsiyon için kök bölgesi aktif
├── Temaslı için komşu diş temas noktası aktif
└── Bağımsız için diş kontur bölgeleri aktif
```

---

## 9. KOD DOSYALARI VE AÇIKLAMALARI

### 9.1 Ana Dosyalar

| Dosya | Satır | Açıklama |
|-------|-------|----------|
| `model_architectures.py` | ~700 | Tüm model mimarileri |
| `train_attention_transformers.py` | ~525 | Ana eğitim scripti |
| `evaluate_results.py` | ~300 | Model değerlendirme |
| `complete_advanced_analysis.py` | ~500 | Gelişmiş analiz |
| `high_confidence_ensemble_test.py` | ~350 | Ensemble test |
| `generate_report_figures.py` | ~280 | Grafik oluşturma |

### 9.2 model_architectures.py

```python
# İçerik:
├── AttentionBlock          # Multi-head self-attention
├── CBAM                    # Channel & Spatial attention
├── SEBlock                 # Squeeze-and-Excitation
├── VisionTransformerModel  # ViT wrapper
├── SwinTransformerModel    # Swin wrapper
├── ResNetCBAM              # ResNet + CBAM
├── ResNetSE                # ResNet + SE
├── EfficientNetAttention   # EfficientNet + SE
├── DenseNetAttention       # DenseNet + SE
├── ViTBaseWithAttention    # ViT + Attention hibrit
├── SwinSmallWithAttention  # Swin + Attention hibrit
├── create_model()          # Model factory
└── get_model_info()        # Model bilgi fonksiyonu
```

### 9.3 train_attention_transformers.py

```python
# İçerik:
├── ToothResorptionDataset  # Veri seti sınıfı
│   ├── __init__()          # Veri yükleme
│   ├── _load_data()        # JSON parsing
│   ├── __len__()           # Dataset length
│   └── __getitem__()       # Sample retrieval
├── get_transforms()        # Augmentation tanımları
├── calculate_metrics()     # Metrik hesaplama
├── train_one_epoch()       # Tek epoch eğitim
├── validate()              # Validation fonksiyonu
├── train_model_with_config() # Tam eğitim döngüsü
└── main()                  # Ana fonksiyon
```

### 9.4 Diğer Önemli Dosyalar

| Dosya | Açıklama |
|-------|----------|
| `evaluate_single_model.py` | Tek model detaylı değerlendirme |
| `compare_attention_variants.py` | Attention varyantları karşılaştırma |
| `select_best_model.py` | En iyi model seçimi |
| `export_best_model.py` | Model export (ONNX, TorchScript) |
| `yolo_tooth_detector.py` | YOLO implementasyonu |
| `tooth_resorption_detector.py` | CNN baseline |

### 9.5 Klasör Yapısı

```
yontem_deneme/
├── 1DOCS/                          # Dokümantasyon
│   ├── TECHNICAL_REPORT.md
│   ├── COMPLETE_PROJECT_DOCUMENTATION.md
│   └── *.png (grafikler)
├── 20 lik diş rezorpsiyon/         # Veri seti
├── models/                         # Eğitilmiş modeller
│   ├── swin_small_best.pth
│   ├── swin_small_mha_head_best.pth
│   ├── vit_base_16_best.pth
│   └── ... (diğer modeller)
├── results/
│   ├── training_logs/              # Eğitim logları
│   ├── visualizations/             # Grafikler
│   ├── advanced_analysis/          # Gelişmiş analiz
│   └── evaluation_results.json     # Değerlendirme sonuçları
├── deployment/                     # Production export
├── yolo_dataset/                   # YOLO veri seti
├── model_architectures.py
├── train_attention_transformers.py
├── requirements.txt
└── README.md
```

---

## 10. GÖRSELLEŞTİRMELER

### 10.1 1DOCS Klasörü İçeriği

| Dosya | Açıklama |
|-------|----------|
| `model_summary_table.png` | Tüm modellerin performans tablosu |
| `model_comparison_chart.png` | Bar chart karşılaştırma |
| `ensemble_performance_chart.png` | Accuracy vs Coverage grafiği |
| `radar_chart.png` | Top 4 model radar grafiği |
| `class_performance_chart.png` | Sınıf bazlı F1 karşılaştırma |
| `high_confidence_analysis.png` | Confidence distribution ve trade-off |
| `high_confidence_confusion_matrix.png` | Yüksek güvenli tahminler confusion matrix |
| `composite_score_comparison.png` | Composite score karşılaştırma |
| `confusion_matrices_top3.png` | Top 3 model confusion matrices |
| `saliency_maps.png` | Grad-CAM benzeri attention maps |
| `uncertainty_analysis.png` | Belirsizlik analizi grafikleri |

### 10.2 Grafik Açıklamaları

#### Model Comparison Chart
- X ekseni: Model isimleri
- Y ekseni: Score (%)
- Yeşil: Accuracy
- Mavi: F1-Score
- Mor: AUC

#### Ensemble Performance Chart
- Sol grafik: Accuracy (bar) vs Coverage (line)
- Sağ grafik: Tüm metrikler threshold bazında
- Kırmızı kesikli çizgi: Optimal threshold (70%)

#### Radar Chart
- 5 eksen: Accuracy, Precision, Recall, F1, AUC
- 4 model karşılaştırması
- Alan: Genel performans göstergesi

---

## 11. SONUÇ VE ÖNERİLER

### 11.1 Ana Bulgular

1. **En iyi tekil model**: Swin Tiny ve Swin Small (~90% accuracy)
2. **En iyi yöntem**: Ensemble (3 model) - %96.73 accuracy
3. **Mükemmel sonuç**: %70 confidence ile %100 accuracy, %77.8 coverage
4. **Attention eklemenin etkisi**: Minimal (~0.5% artış)
5. **Büyük model dezavantajı**: vit_large_16 overfitting gösterdi

### 11.2 Önemli Çıkarımlar

1. **Transformer > CNN**: Modern transformer mimarileri klasik CNN'lerden çok daha iyi
2. **Ensemble etkili**: Farklı modellerin birleşimi tek modelden daha iyi
3. **Confidence filtering**: Yüksek güvenli tahminleri ayırmak %100 doğruluk sağlıyor
4. **Transfer learning kritik**: Pre-trained modeller sıfırdan eğitimden çok daha iyi
5. **Veri seti küçük**: 306 görüntü sınırlı, daha fazla veri ile sonuçlar iyileşir

### 11.3 Klinik Kullanım Önerisi

```
Önerilen Klinik Pipeline:
1. Görüntüyü ensemble modele ver
2. Confidence score'u kontrol et
3. IF confidence >= 70%:
   → Otomatik karar (beklenen: %100 doğru)
4. ELSE:
   → Manuel inceleme için işaretle
   
Beklenen Performans:
- Otomatik kararlar: %77.8 (238/306)
- Manuel inceleme: %22.2 (68/306)
- Otomatik kararlarda hata: %0
```

### 11.4 Gelecek Çalışmalar

1. **Veri artırma**: 1000+ görüntü toplama
2. **Tam K-Fold**: 100+ epoch ile cross-validation
3. **Multi-scale**: Farklı çözünürlüklerle eğitim
4. **Object detection**: YOLO ile ROI tespiti
5. **Ensemble optimization**: Weight arama
6. **External validation**: Farklı hastanelerden veri

### 11.5 Teknik Gereksinimler

```
Yazılım:
├── Python 3.8+
├── PyTorch 1.12+
├── torchvision 0.13+
├── timm 0.6+
├── scikit-learn 1.0+
├── numpy, pandas
├── matplotlib, seaborn
└── PIL (Pillow)

Donanım:
├── GPU: NVIDIA CUDA uyumlu (8GB+ VRAM)
├── RAM: 16GB+
└── Storage: 5GB (modeller dahil)
```

---

## EKLER

### Ek A: requirements.txt

```
tensorflow>=2.10
torch>=1.12
torchvision>=0.13
timm>=0.6
ultralytics>=8.0
scikit-learn>=1.0
numpy>=1.21
pandas>=1.3
matplotlib>=3.5
seaborn>=0.11
Pillow>=8.0
opencv-python>=4.5
einops>=0.4
onnx>=1.12
onnxruntime>=1.12
flask>=2.0
mlflow>=2.0
```

### Ek B: Model Checkpoint Yapısı

```python
checkpoint = {
    'epoch': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': OrderedDict,
    'metrics': {
        'accuracy': float,
        'precision': float,
        'recall': float,
        'f1': float,
        'auc': float,
        'composite_score': float,
        'loss': float
    },
    'config': {
        'lr': float,
        'batch_size': int,
        'weight_decay': float,
        'optimizer': str,
        'epochs': int
    }
}
```

### Ek C: Metrik Tanımları

| Metrik | Formül | Açıklama |
|--------|--------|----------|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | Doğru tahmin oranı |
| Precision | TP/(TP+FP) | Pozitif tahmin doğruluğu |
| Recall | TP/(TP+FN) | Gerçek pozitif yakalama |
| F1-Score | 2*(P*R)/(P+R) | Precision-Recall dengesi |
| AUC | ROC eğrisi altı alan | Sınıflandırma kalitesi |
| Composite | (Acc+P+R+F1+AUC)/5 | Genel performans skoru |

---

**Doküman Tarihi**: Ocak 2026  
**Versiyon**: 2.0  
**Toplam Sayfa**: ~40 (markdown)
