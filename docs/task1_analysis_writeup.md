# Task 1: Data Analysis & Preparation — Write-up

## 1. Dataset Overview

The dataset contains **~1,216 images** across **9 categories** of field check-in types collected by RTV field operations teams. Each category represents a specific agricultural or household installation that field workers photograph during monitoring visits.

| Category | Count | % of Total | Notes |
|----------|-------|------------|-------|
| compost | ~150 | 12.3% | |
| goat-sheep-pen | ~150 | 12.3% | |
| guinea-pig-shelter | **~16** | **1.3%** | Severe minority class |
| liquid-organic | ~150 | 12.3% | |
| organic | ~150 | 12.3% | Visually similar to compost |
| pigsty | ~150 | 12.3% | |
| poultry-house | ~150 | 12.3% | |
| tippytap | ~150 | 12.3% | |
| vsla | ~150 | 12.3% | Non-structural (people in meetings) |

## 2. Key Observations

### 2.1 Class Imbalance (Critical)

The most significant challenge is the **9.4:1 imbalance ratio** between the majority classes (~150 images each) and `guinea-pig-shelter` (~16 images). This is severe enough that a naive classifier could simply never predict this class and still achieve ~98.7% accuracy — a classic accuracy paradox.

**Implication**: We must use metrics that are insensitive to class size (macro F1, per-class recall) rather than raw accuracy to evaluate the model fairly.

### 2.2 Visual Similarity Between Classes

Several categories share significant visual overlap:
- **compost / organic / liquid-organic**: All involve organic matter in outdoor field settings. Distinguishing them requires the model to learn subtle structural differences (e.g., compost is piled, liquid-organic involves containers).
- **Animal shelters**: goat-sheep-pen, guinea-pig-shelter, pigsty, and poultry-house share structural elements (fencing, roofing, enclosures). The model must focus on size, construction material, and animal presence.

**Implication**: A strong pretrained backbone is essential — we need features that capture fine-grained structural differences, not just broad scene categories.

### 2.3 Field Conditions (Noise Sources)

These are real operational images, not curated studio photos:
- Variable lighting (morning/afternoon sun, overcast, indoor shade)
- Inconsistent camera angles and distances
- Background clutter (trees, buildings, people)
- Partial views — structures may be only partially visible
- Motion blur from handheld capture

**Implication**: Augmentation must mirror these real-world variations (rotation, brightness jitter, cropping, blur) rather than exotic transforms that don't occur in practice.

### 2.4 Small Dataset Overall

~1,200 images for a 9-class problem is small by modern standards. Training a deep network from scratch would severely overfit.

**Implication**: Transfer learning from ImageNet is non-negotiable. We use a pretrained YOLO11 backbone and only fine-tune — this gives us strong low-level and mid-level features for free.

## 3. Preprocessing Choices & Rationale

| Step | Choice | Rationale |
|------|--------|-----------|
| **Resize** | 640×640 px | YOLO11 default; sufficient detail for structural features; consistent input |
| **Normalisation** | ImageNet mean/std | Required for pretrained backbone compatibility |
| **Colour mode** | Convert all to RGB | Handles occasional greyscale or RGBA images |
| **Corrupt images** | Flagged and excluded | PIL verify() catches truncated files before training |

### Train / Validation / Test Split

**70% / 15% / 15%** with stratified sampling:

- Stratification ensures proportional representation in every split, even for the 16-image minority class (~11 train / 2 val / 3 test).
- The split is performed **before** any augmentation — this prevents data leakage (a common pitfall where augmented copies of the same image appear in both train and test sets, artificially inflating metrics).
- Random seed is fixed (42) for reproducibility.

## 4. Data Augmentation Strategy

### 4.1 Online Augmentation (during training)

Applied by the YOLO trainer at each epoch — every image is seen differently each time:

| Transform | Parameters | Why |
|-----------|-----------|-----|
| Horizontal flip | p=0.5 | Field photos have no canonical left-right orientation |
| Rotation | ±15° | Compensates for handheld camera tilt |
| HSV jitter | H=0.015, S=0.7, V=0.4 | Handles lighting variation across times of day |
| Scale | 0.5 | Simulates variable distance from subject |
| Translation | 0.1 | Subject not always centred |
| MixUp | α=0.1 | Regularisation — blends images to smooth decision boundaries |
| Random erasing | p=0.3 | Occlusion robustness — objects may be partially hidden |

**Not applied**: Vertical flip (structures have a gravity-dependent orientation), mosaic (designed for detection, not classification).

### 4.2 Offline Augmentation (minority class oversampling)

For `guinea-pig-shelter`, we generate synthetic images **before training** using composed transforms (flip + rotate + colour jitter + crop). The target is **80% of the majority class** count.

**Why 80% and not 100%?** Full equalisation on 16 source images would mean generating ~134 synthetic images from just 16 originals. At that ratio, the model risks memorising augmentation artefacts rather than learning genuine features. 80% is a pragmatic balance — enough to be competitive, not so much that the augmented images dominate.

## 5. Class Imbalance Handling — Three-Layer Strategy

We address imbalance at three different points:

1. **Data level (offline)**: Oversample the minority class via augmentation (see §4.2)
2. **Algorithm level (loss weighting)**: Inverse-frequency class weights — `w_c = N / (C × n_c)` — so the loss penalises minority-class errors proportionally more
3. **Evaluation level**: Report macro F1 (equally weights all classes) alongside weighted F1 and accuracy, so minority-class performance is visible

### Trade-offs Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| SMOTE / synthetic generation | Generates diverse samples | Operates in pixel space; can create unrealistic images | Not used — augmentation is more natural for images |
| Undersampling majority | Simple, fast | Throws away 90% of data on an already small dataset | Not used — data is too scarce |
| Class-weighted loss only | No data manipulation | May not be sufficient for 9.4:1 ratio | Used as one layer, not sole strategy |
| Oversampling + weighting | Combines data and algorithm approaches | More complex | **Chosen** — most robust for this imbalance level |

## 6. Potential Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Guinea-pig-shelter overfits to augmented patterns | Monitor per-class val loss; diverse augmentation pipeline |
| Similar classes confused (compost/organic) | Transfer learning provides fine-grained features; error analysis post-training |
| Data leakage via augmentation | Split is performed before augmentation; val/test never augmented |
| Resolution inconsistency causes artifacts | All images resized to fixed 640×640 with bilinear interpolation |
