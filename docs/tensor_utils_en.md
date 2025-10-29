# Tensor Utils Guide (hodu_utils)

## Overview

`hodu_utils` is a crate that provides data loading and processing utilities for deep learning workflows. It offers an API similar to PyTorch's `torch.utils.data` and supports `no_std` environments.

**Key Components:**

1. **Dataset**: Dataset abstraction and manipulation
2. **DataLoader**: Batch loading, shuffling, and iteration
3. **Sampler**: Sampling strategies
4. **Batch & DataItem**: Type-safe batch processing

**Features:**
- Intuitive PyTorch-style API
- `no_std` support for embedded environments
- Easy custom dataset implementation with `#[derive(Dataset)]` macro
- Type-safe batch processing

## Dataset

### Dataset Trait

Defines the basic interface for datasets.

```rust
pub trait Dataset {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn get(&self, index: usize) -> HoduResult<DataItem>;
}
```

**Implementation Requirements:**
- `len()`: Returns total number of samples
- `get()`: Retrieves sample by index
- `is_empty()`: Automatically implemented (len() == 0)

### TensorDataset

Creates a dataset from tensors.

```rust
use hodu::prelude::*;
use hodu_utils::data::dataset::TensorDataset;

// Data only
let data = Tensor::randn(&[100, 28, 28], DType::F32, Device::Cpu)?;
let dataset = TensorDataset::from_tensor(data);

println!("Dataset length: {}", dataset.len());  // 100

// Get first sample
let first_sample = dataset.get(0)?;  // Shape: [28, 28]
```

**Using Data-Label Pairs:**

```rust
use hodu::prelude::*;
use hodu_utils::data::dataset::TensorDataset;

let data = Tensor::randn(&[100, 28, 28], DType::F32, Device::Cpu)?;
let labels = Tensor::randint(0, 10, &[100], DType::I64, Device::Cpu)?;
let dataset = TensorDataset::from_tensors(data, labels);

// Sample and label returned together
let sample = dataset.get(0)?;  // DataItem::Pair(data, label)
```

**Features:**
- First dimension is the number of samples
- `get(index)` automatically slices the sample at that index
- Returns `DataItem::Pair` if labels exist, `DataItem::Single` otherwise

### Subset

Selects a portion of the full dataset.

```rust
use hodu_utils::data::dataset::Subset;

let indices = vec![0, 2, 4, 6, 8];  // Select even indices only
let subset = Subset::new(dataset, indices);

println!("Subset length: {}", subset.len());  // 5

// subset.get(0) calls original dataset.get(0)
// subset.get(1) calls original dataset.get(2)
```

**Use Cases:**
- Train/validation splits
- Selecting specific classes
- Data sampling

### ConcatDataset

Concatenates multiple datasets into one.

```rust
use hodu_utils::data::dataset::ConcatDataset;

let dataset1 = TensorDataset::from_tensor(data1);  // length 100
let dataset2 = TensorDataset::from_tensor(data2);  // length 50
let dataset3 = TensorDataset::from_tensor(data3);  // length 75

let combined = ConcatDataset::new(vec![dataset1, dataset2, dataset3]);

println!("Total length: {}", combined.len());  // 225

// combined.get(0~99): from dataset1
// combined.get(100~149): from dataset2
// combined.get(150~224): from dataset3
```

**Features:**
- Combines multiple datasets into one logical dataset
- Efficient indexing via cumulative sizes calculation
- Maintains references without copying data

**Use Cases:**
- Combining data from multiple sources
- Merging data augmentation results
- Multi-domain learning

### random_split

Randomly splits a dataset into training and validation sets.

```rust
use hodu_utils::data::dataset::random_split;

let dataset = TensorDataset::from_tensors(data, labels);

// 80% for training, 20% for validation
let (train_dataset, val_dataset) = random_split(dataset, 0.8, Some(42));

println!("Train size: {}", train_dataset.len());  // 80
println!("Val size: {}", val_dataset.len());      // 20
```

**Parameters:**
- `dataset`: Dataset to split (must implement Clone)
- `train_size`: Training data ratio (0.0 ~ 1.0)
- `seed`: Seed value for reproducibility (Option<u64>)

**Features:**
- Uses Fisher-Yates shuffle algorithm
- Splits indices without copying data
- Reproducible splits with seed value

**Example: Various Split Ratios:**

```rust
// 70% / 30% split
let (train, val) = random_split(dataset.clone(), 0.7, Some(42));

// 90% / 10% split
let (train, val) = random_split(dataset.clone(), 0.9, Some(42));

// Different split each time without seed
let (train, val) = random_split(dataset, 0.8, None);
```

## DataLoader

Loads and iterates over data in batches.

### Basic Usage

```rust
use hodu_utils::data::dataloader::DataLoader;

let dataset = TensorDataset::from_tensors(data, labels);
let mut loader = DataLoader::new(dataset, 32);  // Batch size 32

// Iterate over batches
while let Some(batch) = loader.next_batch()? {
    match batch {
        Batch::Pair(data, labels) => {
            println!("Batch data shape: {:?}", data.get_shape());      // [32, ...]
            println!("Batch labels shape: {:?}", labels.get_shape());  // [32]
        }
        _ => {}
    }
}
```

**Default Settings:**
- `batch_size`: 32
- `shuffle`: false
- `drop_last`: false
- `collate_fn`: default_collate

### DataLoader::builder()

Configure more options.

```rust
let mut loader = DataLoader::builder(dataset)
    .batch_size(32)           // Batch size
    .shuffle(true)            // Shuffle every epoch
    .drop_last(true)          // Drop last incomplete batch
    .seed(42)                 // Seed for reproducibility
    .build();

println!("Number of batches: {}", loader.num_batches());
```

**Builder Methods:**

| Method | Description | Default |
|--------|-------------|---------|
| `batch_size(usize)` | Set batch size | 1 |
| `shuffle(bool)` | Whether to shuffle every epoch | false |
| `drop_last(bool)` | Drop last incomplete batch | false |
| `seed(u64)` | Set random seed | None |
| `collate_fn(CollateFn)` | Custom collate function | default_collate |

### Iteration Patterns

#### Pattern 1: while let Loop

```rust
let mut loader = DataLoader::new(dataset, 32);

while let Some(batch) = loader.next_batch()? {
    // Process batch
    if let Batch::Pair(data, labels) = batch {
        let predictions = model.forward(&data)?;
        let loss = criterion(&predictions, &labels)?;
    }
}
```

#### Pattern 2: Iterator Interface

```rust
let mut loader = DataLoader::new(dataset, 32);

for batch_result in loader.iter_batches() {
    let batch = batch_result?;
    // Process batch
}
```

**Iterator Advantages:**
- Use Rust's standard Iterator methods
- Compatible with `map`, `filter`, `collect`, etc.
- Functional programming style

### Epoch Management

```rust
let mut loader = DataLoader::builder(dataset)
    .batch_size(32)
    .shuffle(true)
    .seed(42)
    .build();

for epoch in 0..10 {
    println!("Epoch {}", epoch);

    // Iteration
    for batch_result in loader.iter_batches() {
        let batch = batch_result?;
        // Training logic
    }

    // Reset for next epoch
    // Automatically re-shuffles if shuffle=true
    loader.reset();
}
```

**reset() Behavior:**
- Resets current index to 0
- Increments epoch counter
- Re-shuffles with new seed if `shuffle=true`
  - New seed = original seed + epoch number

### Batch Size and drop_last

```rust
// Example: Dataset size is 100

// drop_last=false (default)
let loader1 = DataLoader::builder(dataset.clone())
    .batch_size(32)
    .drop_last(false)
    .build();
// Batches: [32, 32, 32, 4]  -> 4 batches total

// drop_last=true
let loader2 = DataLoader::builder(dataset)
    .batch_size(32)
    .drop_last(true)
    .build();
// Batches: [32, 32, 32]  -> 3 batches (last 4 samples dropped)
```

**When to Set drop_last=true:**
- Using Batch Normalization (small batches are unstable)
- All batches must have the same size
- Maintaining consistent batch size in distributed training

### Custom Collate Function

Customize batch creation logic.

```rust
use hodu_utils::data::batch::{Batch, DataItem};
use hodu_utils::data::dataloader::CollateFn;

// Define custom collate function
fn my_collate(items: Vec<DataItem>) -> HoduResult<Batch> {
    // Example: Normalize each sample while creating batch
    let tensors: Vec<Tensor> = items
        .into_iter()
        .map(|item| match item {
            DataItem::Single(t) => {
                // Normalize: (x - mean) / std
                t.sub_scalar(0.5)?.div_scalar(0.5)
            }
            _ => panic!("Unexpected item type"),
        })
        .collect::<Result<Vec<_>, _>>()?;

    let refs: Vec<&Tensor> = tensors.iter().collect();
    let batched = Tensor::stack(&refs, 0)?;
    Ok(Batch::Single(batched))
}

// Apply to DataLoader
let loader = DataLoader::builder(dataset)
    .batch_size(32)
    .collate_fn(my_collate)
    .build();
```

**Use Cases:**
- Dynamic padding (variable-length sequences)
- Data augmentation
- Normalization
- Special batch formats

## Sampler

Controls the sampling order of the dataset.

### SequentialSampler

Samples sequentially.

```rust
use hodu_utils::data::sampler::SequentialSampler;

let sampler = SequentialSampler::new(100);

// Iterate in order: 0, 1, 2, ..., 99
for idx in sampler.iter() {
    println!("{}", idx);
}
```

**Features:**
- Simplest sampler
- Indices from 0 to length-1 sequentially
- Reproducible order

### RandomSampler

Samples randomly.

```rust
use hodu_utils::data::sampler::RandomSampler;

// Shuffle 100 indices randomly
let sampler = RandomSampler::new(100, Some(42));

for idx in sampler.iter() {
    println!("{}", idx);  // Random order: 73, 21, 5, ...
}
```

**Shuffling Specific Indices:**

```rust
let indices = vec![0, 5, 10, 15, 20];
let sampler = RandomSampler::with_indices(indices, Some(42));

for idx in sampler.iter() {
    println!("{}", idx);  // 5, 20, 0, 15, 10, etc.
}
```

**Features:**
- Uses Fisher-Yates shuffle algorithm
- Reproducible with seed value
- Samples each index exactly once (no duplicates)

### SubsetSampler

Samples only specified indices.

```rust
use hodu_utils::data::sampler::SubsetSampler;

let indices = vec![0, 2, 4, 6, 8];  // Even indices only
let sampler = SubsetSampler::new(indices);

for idx in sampler.iter() {
    println!("{}", idx);  // 0, 2, 4, 6, 8
}
```

**Use Cases:**
- Sampling specific classes
- Specifying validation set indices
- Defining data subsets

### BatchSampler

Groups a sampler into batches.

```rust
use hodu_utils::data::sampler::{SequentialSampler, BatchSampler};

let sampler = SequentialSampler::new(100);
let batch_sampler = BatchSampler::new(sampler, 32, false);

for batch_indices in batch_sampler.batches() {
    println!("Batch: {:?}", batch_indices);
    // [0, 1, ..., 31]
    // [32, 33, ..., 63]
    // [64, 65, ..., 95]
    // [96, 97, 98, 99]  (last batch has 4)
}
```

**drop_last Effect:**

```rust
// drop_last=false
let batch_sampler1 = BatchSampler::new(sampler.clone(), 32, false);
// Batches: [32, 32, 32, 4]

// drop_last=true
let batch_sampler2 = BatchSampler::new(sampler, 32, true);
// Batches: [32, 32, 32]  (last 4 dropped)
```

**Notes:**
- BatchSampler returns array of arrays of indices
- Can be used separately from DataLoader
- Useful for implementing custom batch loading logic

## Batch & DataItem

Enums for type-safe data representation.

### DataItem

Represents a single sample.

```rust
use hodu_utils::data::batch::DataItem;

// Single tensor (e.g., image only)
let item1 = DataItem::single(image_tensor);

// Data-label pair (e.g., image + class)
let item2 = DataItem::pair(image_tensor, label_tensor);

// Multiple tensors (e.g., image + segmentation mask + bounding box)
let item3 = DataItem::multiple(vec![image, mask, bbox]);
```

**Enum Definition:**

```rust
pub enum DataItem {
    Single(Tensor),
    Pair(Tensor, Tensor),
    Multiple(Vec<Tensor>),
}
```

### Batch

Represents batch data.

```rust
use hodu_utils::data::batch::Batch;

// Process batch with pattern matching
match batch {
    Batch::Single(tensor) => {
        // Single tensor batch
        println!("Batch shape: {:?}", tensor.get_shape());  // [batch_size, ...]
    }
    Batch::Pair(data, labels) => {
        // Data-label batch
        println!("Data shape: {:?}", data.get_shape());      // [batch_size, ...]
        println!("Labels shape: {:?}", labels.get_shape());  // [batch_size]
    }
    Batch::Multiple(tensors) => {
        // Multiple tensors batch
        for (i, tensor) in tensors.iter().enumerate() {
            println!("Tensor {} shape: {:?}", i, tensor.get_shape());
        }
    }
}
```

**Conversion Methods:**

```rust
// Convert Batch::Pair to tuple
if let Some((data, labels)) = batch.into_pair() {
    // Use data and labels
    let predictions = model.forward(&data)?;
}

// Convert Batch::Single to tensor
if let Some(tensor) = batch.into_single() {
    // Use tensor
}

// Convert Batch::Multiple to Vec
if let Some(tensors) = batch.into_multiple() {
    // Use tensors vector
}
```

**Features:**
- Type-safe: Compile-time batch type checking
- Explicit: Clear what data is included
- Flexible: Supports various data structures

### default_collate

Understanding the default collate function behavior.

```rust
use hodu_utils::data::dataloader::default_collate;

// Convert DataItem::Single to Batch::Single
let items = vec![
    DataItem::single(tensor1),  // [28, 28]
    DataItem::single(tensor2),  // [28, 28]
    DataItem::single(tensor3),  // [28, 28]
];
let batch = default_collate(items)?;
// Batch::Single([3, 28, 28])  <- Combined with stack(dim=0)

// Convert DataItem::Pair to Batch::Pair
let items = vec![
    DataItem::pair(data1, label1),
    DataItem::pair(data2, label2),
    DataItem::pair(data3, label3),
];
let batch = default_collate(items)?;
// Batch::Pair([3, ...], [3])  <- Each stacked separately
```

**default_collate Behavior:**
1. Verify all items are the same type
2. Collect each tensor into a list
3. Combine with `Tensor::stack(&refs, 0)` at dim=0
4. Return appropriate Batch type

## Dataset Derive Macro

Easily implement custom datasets using `#[derive(Dataset)]`.

### Basic Usage

```rust
use hodu_utils::data::dataset::Dataset;
use hodu_utils::data::batch::DataItem;

#[derive(Dataset)]
pub struct ImageDataset {
    images: Vec<Tensor>,
    labels: Vec<i32>,
}

impl ImageDataset {
    pub fn new(images: Vec<Tensor>, labels: Vec<i32>) -> Self {
        Self { images, labels }
    }

    // Implement len() and get() for Dataset trait
    fn len(&self) -> usize {
        self.images.len()
    }

    fn get(&self, index: usize) -> HoduResult<DataItem> {
        let image = self.images[index].clone();
        let label = Tensor::full(&[], self.labels[index] as f32)?;
        Ok(DataItem::pair(image, label))
    }
}
```

**Code Auto-Generated by Macro:**

```rust
impl Dataset for ImageDataset {
    fn len(&self) -> usize {
        self.len()  // Calls user-defined len()
    }

    fn get(&self, index: usize) -> HoduResult<DataItem> {
        self.get(index)  // Calls user-defined get()
    }
}
```

### Advanced Example: Image Augmentation Dataset

```rust
use hodu_utils::data::dataset::Dataset;
use hodu_utils::data::batch::DataItem;

#[derive(Dataset)]
pub struct AugmentedDataset {
    base_dataset: TensorDataset,
    augment: bool,
}

impl AugmentedDataset {
    pub fn new(base_dataset: TensorDataset, augment: bool) -> Self {
        Self { base_dataset, augment }
    }

    fn len(&self) -> usize {
        self.base_dataset.len()
    }

    fn get(&self, index: usize) -> HoduResult<DataItem> {
        let item = self.base_dataset.get(index)?;

        if !self.augment {
            return Ok(item);
        }

        // Apply data augmentation
        match item {
            DataItem::Pair(data, label) => {
                // Random flip
                let augmented = if rand::random::<bool>() {
                    data.flip(vec![2])?  // Horizontal flip
                } else {
                    data
                };

                // Add random noise
                let noise = Tensor::randn_like(&augmented, 0.0, 0.1)?;
                let augmented = augmented.add(&noise)?;

                Ok(DataItem::pair(augmented, label))
            }
            _ => Ok(item),
        }
    }
}
```

## Practical Examples

### Example 1: Basic Training Loop

```rust
use hodu::prelude::*;
use hodu_utils::data::dataset::TensorDataset;
use hodu_utils::data::dataloader::DataLoader;
use hodu_utils::data::batch::Batch;

// Prepare data
let data = Tensor::randn(&[1000, 28, 28], DType::F32, Device::Cpu)?;
let labels = Tensor::randint(0, 10, &[1000], DType::I64, Device::Cpu)?;

// Create dataset and DataLoader
let dataset = TensorDataset::from_tensors(data, labels);
let mut loader = DataLoader::builder(dataset)
    .batch_size(32)
    .shuffle(true)
    .seed(42)
    .build();

// Training loop
for epoch in 0..10 {
    println!("Epoch {}", epoch);

    for batch_result in loader.iter_batches() {
        let batch = batch_result?;

        if let Batch::Pair(data, labels) = batch {
            // Forward pass
            let predictions = model.forward(&data)?;
            let loss = criterion(&predictions, &labels)?;

            // Backward pass
            loss.backward()?;

            // Optimizer step
            optimizer.step()?;
            optimizer.zero_grad()?;
        }
    }

    loader.reset();
}
```

### Example 2: Train/Validation Split

```rust
use hodu_utils::data::dataset::{TensorDataset, random_split};
use hodu_utils::data::dataloader::DataLoader;

// Prepare data
let data = Tensor::randn(&[1000, 784], DType::F32, Device::Cpu)?;
let labels = Tensor::randint(0, 10, &[1000], DType::I64, Device::Cpu)?;

// Create dataset and split
let dataset = TensorDataset::from_tensors(data, labels);
let (train_dataset, val_dataset) = random_split(dataset, 0.8, Some(42));

println!("Train size: {}", train_dataset.len());  // 800
println!("Val size: {}", val_dataset.len());      // 200

// Create DataLoaders
let mut train_loader = DataLoader::builder(train_dataset)
    .batch_size(32)
    .shuffle(true)
    .drop_last(true)
    .seed(42)
    .build();

let mut val_loader = DataLoader::builder(val_dataset)
    .batch_size(32)
    .shuffle(false)  // Don't shuffle validation set
    .build();

// Training and validation
for epoch in 0..10 {
    // Training
    train!();
    let mut train_loss = 0.0;
    let mut train_count = 0;

    for batch_result in train_loader.iter_batches() {
        let batch = batch_result?;
        if let Batch::Pair(data, labels) = batch {
            let predictions = model.forward(&data)?;
            let loss = criterion(&predictions, &labels)?;

            loss.backward()?;
            optimizer.step()?;
            optimizer.zero_grad()?;

            train_loss += loss.to_scalar::<f32>()?;
            train_count += 1;
        }
    }
    train_loader.reset();

    // Validation
    eval!();
    let mut val_loss = 0.0;
    let mut val_count = 0;

    for batch_result in val_loader.iter_batches() {
        let batch = batch_result?;
        if let Batch::Pair(data, labels) = batch {
            let predictions = model.forward(&data)?;
            let loss = criterion(&predictions, &labels)?;

            val_loss += loss.to_scalar::<f32>()?;
            val_count += 1;
        }
    }
    val_loader.reset();

    println!("Epoch {}: Train Loss = {:.4}, Val Loss = {:.4}",
             epoch,
             train_loss / train_count as f32,
             val_loss / val_count as f32);
}
```

### Example 3: Custom Dataset with Augmentation

```rust
use hodu_utils::data::dataset::Dataset;
use hodu_utils::data::batch::DataItem;
use hodu_utils::data::dataloader::DataLoader;

#[derive(Dataset)]
pub struct CustomDataset {
    data_paths: Vec<String>,
    labels: Vec<i32>,
    transform: bool,
}

impl CustomDataset {
    pub fn new(data_paths: Vec<String>, labels: Vec<i32>, transform: bool) -> Self {
        Self { data_paths, labels, transform }
    }

    fn len(&self) -> usize {
        self.data_paths.len()
    }

    fn get(&self, index: usize) -> HoduResult<DataItem> {
        // Load data from file (actual implementation needed)
        let data = load_data_from_file(&self.data_paths[index])?;

        // Apply transform
        let data = if self.transform {
            apply_augmentation(data)?
        } else {
            data
        };

        let label = Tensor::full(&[], self.labels[index] as f32)?;
        Ok(DataItem::pair(data, label))
    }
}

// Usage
let paths = vec![/* ... */];
let labels = vec![/* ... */];

let train_dataset = CustomDataset::new(paths.clone(), labels.clone(), true);
let val_dataset = CustomDataset::new(paths, labels, false);

let train_loader = DataLoader::new(train_dataset, 32);
let val_loader = DataLoader::new(val_dataset, 32);
```

### Example 4: Multi-Source Data Combining

```rust
use hodu_utils::data::dataset::{TensorDataset, ConcatDataset};

// Data from multiple sources
let dataset1 = TensorDataset::from_tensors(data1, labels1);  // Source A
let dataset2 = TensorDataset::from_tensors(data2, labels2);  // Source B
let dataset3 = TensorDataset::from_tensors(data3, labels3);  // Source C

// Combine all data
let combined = ConcatDataset::new(vec![dataset1, dataset2, dataset3]);

println!("Total samples: {}", combined.len());

// Train with combined data
let loader = DataLoader::builder(combined)
    .batch_size(64)
    .shuffle(true)
    .build();
```

### Example 5: Handling Imbalanced Data

```rust
use hodu_utils::data::sampler::{RandomSampler, SubsetSampler};
use hodu_utils::data::dataset::Subset;

// Collect indices by class
let mut class_0_indices = Vec::new();
let mut class_1_indices = Vec::new();

for i in 0..dataset.len() {
    let item = dataset.get(i)?;
    if let DataItem::Pair(_, label) = item {
        let class = label.to_scalar::<i32>()?;
        if class == 0 {
            class_0_indices.push(i);
        } else {
            class_1_indices.push(i);
        }
    }
}

// Oversample minority class
let target_size = class_0_indices.len().max(class_1_indices.len());

// Class 0 subset (use as is)
let class_0_subset = Subset::new(dataset.clone(), class_0_indices);

// Class 1 subset (repeat sampling to match size)
let mut balanced_class_1_indices = class_1_indices.clone();
while balanced_class_1_indices.len() < target_size {
    balanced_class_1_indices.extend(&class_1_indices);
}
balanced_class_1_indices.truncate(target_size);
let class_1_subset = Subset::new(dataset, balanced_class_1_indices);

// Create balanced dataset
let balanced_dataset = ConcatDataset::new(vec![class_0_subset, class_1_subset]);

let loader = DataLoader::builder(balanced_dataset)
    .batch_size(32)
    .shuffle(true)
    .build();
```

## no_std Support

`hodu_utils` supports `no_std` environments.

### Cargo.toml Configuration

```toml
# no_std environment (disable default features)
[dependencies]
hodu_utils = { version = "0.1.9", default-features = false }

# std environment (enable standard features)
[dependencies]
hodu_utils = { version = "0.1.9", features = ["std"] }

# Add serde support
[dependencies]
hodu_utils = { version = "0.1.9", features = ["std", "serde"] }
```

### Feature Flags

| Flag | Description | Default | Dependencies |
|------|-------------|---------|--------------|
| `std` | Standard library support | ✅ | - |
| `serde` | Serialization/Deserialization | ❌ | hodu_core/serde |

### no_std Usage Example

```rust
#![no_std]

extern crate alloc;
use alloc::vec::Vec;

use hodu_utils::data::dataset::TensorDataset;
use hodu_utils::data::dataloader::DataLoader;

// Use the same way in no_std environment
let dataset = TensorDataset::from_tensor(data);
let loader = DataLoader::new(dataset, 32);
```

**Limitations:**
- Cannot use `std::io` related features
- Some random generation features are limited
- Requires `alloc` crate (dynamic memory allocation)

## Cautions

### 1. Calling DataLoader's reset()

```rust
// Incorrect usage: Reusing without reset()
let mut loader = DataLoader::new(dataset, 32);

// First epoch
for batch in loader.iter_batches() { /* ... */ }

// Second epoch - no batches returned!
for batch in loader.iter_batches() { /* ... */ }  // ❌

// Correct usage: Call reset()
let mut loader = DataLoader::new(dataset, 32);

for epoch in 0..10 {
    for batch in loader.iter_batches() { /* ... */ }
    loader.reset();  // ✅ Reset for next epoch
}
```

### 2. drop_last and Batch Normalization

```rust
// Recommended drop_last=true when using Batch Normalization
let loader = DataLoader::builder(dataset)
    .batch_size(32)
    .drop_last(true)  // Drop last small batch
    .build();

// Reason: Statistics calculation is unstable with too small batches
// Example: Batch size of 1 or 2 makes variance calculation meaningless
```

### 3. Shuffle Seed Changes Per Epoch

```rust
let mut loader = DataLoader::builder(dataset)
    .batch_size(32)
    .shuffle(true)
    .seed(42)
    .build();

// Epoch 0: Shuffle with seed 42
for batch in loader.iter_batches() { /* ... */ }
loader.reset();

// Epoch 1: Shuffle with seed 43 (42 + 1)
for batch in loader.iter_batches() { /* ... */ }
loader.reset();

// Epoch 2: Shuffle with seed 44 (42 + 2)
// ...
```

**For Reproducibility:**
- Use the same seed value
- Different shuffle applied per epoch but reproducible

### 4. Clone Requirements

```rust
// random_split requires Clone
let dataset = TensorDataset::from_tensors(data, labels);
let (train, val) = random_split(dataset, 0.8, Some(42));  // ✅

// ConcatDataset transfers ownership
let dataset1 = TensorDataset::from_tensor(data1);
let dataset2 = TensorDataset::from_tensor(data2);
let combined = ConcatDataset::new(vec![dataset1, dataset2]);
// dataset1, dataset2 can't be used afterwards

// Clone in advance for reuse
let dataset1 = TensorDataset::from_tensor(data1);
let dataset1_clone = dataset1.clone();
let combined = ConcatDataset::new(vec![dataset1, dataset2]);
// dataset1_clone is still usable
```

### 5. DataItem Type Consistency

```rust
// default_collate requires all items to be the same type

// ❌ Incorrect usage
let items = vec![
    DataItem::single(tensor1),
    DataItem::pair(tensor2, label),  // Type mismatch!
];
let batch = default_collate(items)?;  // Panic!

// ✅ Correct usage
let items = vec![
    DataItem::pair(data1, label1),
    DataItem::pair(data2, label2),
    DataItem::pair(data3, label3),
];
let batch = default_collate(items)?;  // OK
```

### 6. Index Range Checking

```rust
// Dataset::get() should perform range checking

impl MyDataset {
    fn get(&self, index: usize) -> HoduResult<DataItem> {
        // ✅ Range check
        if index >= self.len() {
            return Err(HoduError::InternalError(
                format!("Index {} out of bounds", index)
            ));
        }

        Ok(DataItem::single(self.data[index].clone()))
    }
}
```

## Comparison Tables

### Dataset Creation

| Function | Input | Output | Use Case |
|----------|-------|--------|----------|
| `TensorDataset::from_tensor()` | Single tensor | DataItem::Single | Unsupervised learning, autoencoders |
| `TensorDataset::from_tensors()` | Data, labels | DataItem::Pair | Supervised learning, classification/regression |
| `Subset::new()` | Dataset, indices | Subset | Data splitting, subset selection |
| `ConcatDataset::new()` | Vec<Dataset> | ConcatDataset | Multi-source combining |
| `random_split()` | Dataset, ratio | (Subset, Subset) | Train/validation split |

### DataLoader Configuration

| Method | Parameter | Effect |
|--------|-----------|--------|
| `batch_size()` | usize | Set batch size |
| `shuffle()` | bool | Whether to shuffle every epoch |
| `drop_last()` | bool | Drop last incomplete batch |
| `seed()` | u64 | Set random seed |
| `collate_fn()` | CollateFn | Custom collate function |

### Sampler Types

| Sampler | Order | Reproducibility | Use Case |
|---------|-------|-----------------|----------|
| `SequentialSampler` | Sequential (0, 1, 2, ...) | Always same | Evaluation, debugging |
| `RandomSampler` | Random (shuffled) | Controlled by seed | Training, data augmentation |
| `SubsetSampler` | Specified index order | Always same | Specific sample selection |
| `BatchSampler` | Grouped by batch | Depends on base sampler | Custom batch logic |

## Summary

| Category | Component | Key Function |
|----------|-----------|--------------|
| **Dataset** | TensorDataset | Create dataset from tensors |
| | Subset | Select portion of dataset |
| | ConcatDataset | Combine multiple datasets |
| | random_split | Train/validation split |
| **DataLoader** | builder | Flexible configuration |
| | iter_batches | Iterator interface |
| | reset | Epoch management |
| **Sampler** | Sequential | Sequential sampling |
| | Random | Random sampling |
| | Subset | Index-based sampling |
| | Batch | Batch grouping |
| **Types** | DataItem | Single sample representation |
| | Batch | Batch data representation |

**Core Workflow:**
1. Create Dataset → 2. Configure DataLoader → 3. Iterate Batches → 4. Train/Validate

**Best Practices:**
- Use `shuffle=true` for training, `shuffle=false` for validation
- Set `drop_last=true` when using Batch Normalization
- Call `loader.reset()` every epoch
- Set seed value for reproducibility
- Use `#[derive(Dataset)]` for custom data
