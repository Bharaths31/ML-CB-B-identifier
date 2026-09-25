# Core Architecture Components Analysis

This document details the location, usage, and purpose of key structural and framework components within the Cattle & Buffalo Breed Classifier project.

---

## 1. EfficientNet-Lite2

### Where and How It Is Used
**EfficientNet-Lite2** serves as the **core visual feature extraction backbone** for the architecture.

- **Feature Extraction & Staging:** Raw input images ($260 \times 260 \times 3$) pass through stages 0 to 3 of EfficientNet-Lite2, outputting 88-channel feature maps.
- **CBAM Attention Insertion:** An attention block (CBAM) is inserted immediately after Stage 3 (`CBAM_AFTER_STAGE = 3`).
- **Remaining Stages & Pooling:** The attended features flow through stages 4 to 6 and the conv head, resulting in a 1280-dimensional feature vector via adaptive average pooling.
- **Multi-Head Routing:** These 1280-d features are fed into three classification heads:
  1. `binary_head`: Species classifier (Cattle vs. Buffalo)
  2. `cattle_head`: Identifies 57 cattle breeds
  3. `buffalo_head`: Identifies 18 buffalo breeds

### Relevant File Locations
- **`src/efficientnet_lite.py`**: Full PyTorch implementation tailored for edge inference.
- **`src/model.py`**: Instantiates `EfficientNetLite(arch="lite2")` inside `BreedClassifier`.
- **`src/config.py`**: Defines input dimensions, stage channel maps, and backbone weight paths.
- **`efficientnet_lite2.pth`**: Pretrained ImageNet weights loaded at training startup.

### Purpose & Design Rationale
1. **Mobile & Edge Device Optimization:** Replaces Swish with ReLU6 and eliminates standard SE-blocks, enabling efficient INT8 quantization for sub-100ms CPU inference on mobile devices.
2. **Resolution & Latency Sweet Spot:** Operates efficiently on $260 \times 260$ resolution to capture subtle indigenous breed features (horn orientation, coat patterns) without heavy computational overhead.

---

## 2. CBAM (Convolutional Block Attention Module)

### Where and How It Is Used
**CBAM** is a lightweight attention mechanism integrated directly into the backbone's feature stream.

- **Mid-Network Insertion:** By default, it receives intermediate 88-channel feature maps from Stage 3 of the EfficientNet-Lite2 backbone.
- **Dual Attention Mechanism:** 
  1. **Channel Attention:** Identifies "what" is important (e.g., feature maps responding strongly to coat patterns).
  2. **Spatial Attention:** Identifies "where" it is important (e.g., focusing on the animal's hump or horns while suppressing background noise).
- **Identity Init:** Initialized to act as an identity transformation initially (`CBAM_IDENTITY_INIT = True`), allowing the pre-trained backbone to remain stable while the attention layers slowly learn.

### Relevant File Locations
- **`src/cbam.py`**: Contains the implementations of `ChannelAttention`, `SpatialAttention`, and the combined `CBAM` module.
- **`src/model.py`**: Initializes CBAM inside `BreedClassifier` and explicitly calls it between backbone stages: `feats = self.attention(feats)`.
- **`src/config.py`**: Configures the insertion point (`CBAM_AFTER_STAGE`) and initialization behavior (`CBAM_IDENTITY_INIT`).

### Purpose & Design Rationale
- **Fine-Grained Classification:** Many cattle breeds look nearly identical in shape and are only distinguished by localized features like dewlap size, head shape, or subtle horn angles. CBAM teaches the network to explicitly look for these localized traits, dramatically improving fine-grained accuracy over a standard backbone.

---

## 3. PyTorch (`torch`)

### Where and How It Is Used
**PyTorch** is the foundational deep learning framework powering the entire training and inference logic.

- **Model Definition:** All architectural components inherit from `torch.nn.Module`. Network parameters are defined as `torch.Tensor`s.
- **Optimization & Loss:** Uses `torch.optim` (like AdamW), CrossEntropy losses, and auxiliary losses (e.g., Supervised Contrastive Loss) for backpropagation.
- **Mixed Precision:** Utilizes `torch.amp` (Automatic Mixed Precision) during training on CUDA devices for faster, memory-efficient updates.
- **State Serialization:** Saves and loads raw model weights via `torch.save` and `torch.load` into `.pt` checkpoint files.

### Relevant File Locations
- **`src/model.py`, `src/cbam.py`, `src/traits.py`**: Model building blocks utilizing `torch.nn` and `torch.nn.functional`.
- **`src/train.py`**: Orchestrates the core PyTorch training loop (forward pass, loss backward, optimizer step, EMA tracking, checkpointing).
- **`src/export.py`**: Converts raw PyTorch models (`.pt`) into TorchScript and ONNX graphs for platform-agnostic deployment.
- **`test_model.py`**: Uses PyTorch for CPU/GPU inference testing before deployment.

### Purpose & Design Rationale
- **Dynamic Computational Graphs:** PyTorch provides absolute control over custom training logic (e.g., phase-based freezing, knowledge distillation, mixup/cutmix, and multi-head routing).
- **Ecosystem Portability:** Serves as the origin point that safely traces/exports to ONNX and Mobile INT8 quantization logic (QAT).

---

## 4. Torchvision (`torchvision`)

### Where and How It Is Used
**Torchvision** is PyTorch's primary vision library, utilized exclusively for the image data pipeline, preprocessing, and augmentation.

- **Dataset Handling:** Uses `torchvision.datasets.ImageFolder` (via wrappers) to automatically infer class labels from folder structures.
- **Augmentation Pipeline (Train):** Applies dynamic geometric and color transformations via `transforms.Compose`, including:
  - `RandomResizedCrop` (with custom aspect ratios to preserve body proportions).
  - `RandomHorizontalFlip`
  - `ColorJitter` (carefully tuned to preserve breed-defining coat colors).
- **Evaluation Pipeline (Test):** Standardizes inputs for inference using a strict sequence of shortest-side `Resize`, `CenterCrop`, `ToTensor`, and ImageNet `Normalize`.

### Relevant File Locations
- **`src/data_pipeline.py`**: The central hub for all `torchvision.transforms`, dataset instantiation, caching logic, and `DataLoader` creation.
- **`src/config.py`**: Configures the exact strengths and toggles for torchvision augmentations (`RRC_SCALE`, `COLOR_JITTER_HUE`, `AUG_HORIZONTAL_FLIP`).
- **`test_model.py`**: Directly uses torchvision transforms to cleanly preprocess user-uploaded images identically to the validation set.

### Purpose & Design Rationale
- **Deterministic and Robust Augmentation:** Provides highly-optimized, C++ backed image augmentations on the CPU, allowing the data-loading workers to saturate the GPU without becoming a bottleneck.
- **Breed-Aware Processing:** Allows the framework to dynamically toggle `torchvision` augmentations on a per-breed basis (e.g., disabling `ColorJitter` dynamically for specific coat-color dependent breeds).
