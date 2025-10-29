# Tensor Utils 가이드 (hodu_utils)

## 개요

`hodu_utils`는 딥러닝 워크플로우에 필요한 데이터 로딩 및 처리 유틸리티를 제공하는 크레이트입니다. PyTorch의 `torch.utils.data`와 유사한 API를 제공하며, `no_std` 환경을 지원합니다.

**주요 컴포넌트:**

1. **Dataset**: 데이터셋 추상화 및 조작
2. **DataLoader**: 배치 로딩, 셔플링, 이터레이션
3. **Sampler**: 샘플링 전략
4. **Batch & DataItem**: 타입 안전한 배치 처리

**특징:**
- PyTorch 스타일 API로 직관적인 사용
- `no_std` 지원으로 임베디드 환경에서도 사용 가능
- `#[derive(Dataset)]` 매크로로 커스텀 데이터셋 쉽게 구현
- 타입 안전한 배치 처리

## Dataset

### Dataset 트레이트

데이터셋의 기본 인터페이스를 정의합니다.

```rust
pub trait Dataset {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn get(&self, index: usize) -> HoduResult<DataItem>;
}
```

**구현 요구사항:**
- `len()`: 데이터셋의 전체 샘플 수 반환
- `get()`: 인덱스로 샘플 가져오기
- `is_empty()`: 자동 구현 (len() == 0)

### TensorDataset

텐서로부터 데이터셋을 생성합니다.

```rust
use hodu::prelude::*;
use hodu_utils::data::dataset::TensorDataset;

// 데이터만 있는 경우
let data = Tensor::randn(&[100, 28, 28], DType::F32, Device::Cpu)?;
let dataset = TensorDataset::from_tensor(data);

println!("Dataset length: {}", dataset.len());  // 100

// 첫 번째 샘플 가져오기
let first_sample = dataset.get(0)?;  // Shape: [28, 28]
```

**데이터-레이블 쌍 사용:**

```rust
use hodu::prelude::*;
use hodu_utils::data::dataset::TensorDataset;

let data = Tensor::randn(&[100, 28, 28], DType::F32, Device::Cpu)?;
let labels = Tensor::randint(0, 10, &[100], DType::I64, Device::Cpu)?;
let dataset = TensorDataset::from_tensors(data, labels);

// 샘플과 레이블이 함께 반환됨
let sample = dataset.get(0)?;  // DataItem::Pair(data, label)
```

**특징:**
- 첫 번째 차원이 샘플 수
- `get(index)`는 해당 인덱스의 샘플을 자동으로 slicing
- 레이블이 있으면 `DataItem::Pair`, 없으면 `DataItem::Single` 반환

### Subset

전체 데이터셋의 일부만 선택합니다.

```rust
use hodu_utils::data::dataset::Subset;

let indices = vec![0, 2, 4, 6, 8];  // 짝수 인덱스만 선택
let subset = Subset::new(dataset, indices);

println!("Subset length: {}", subset.len());  // 5

// subset.get(0)은 원본 dataset.get(0)을 호출
// subset.get(1)은 원본 dataset.get(2)를 호출
```

**사용 사례:**
- 학습/검증 분할
- 특정 클래스만 선택
- 데이터 샘플링

### ConcatDataset

여러 데이터셋을 하나로 연결합니다.

```rust
use hodu_utils::data::dataset::ConcatDataset;

let dataset1 = TensorDataset::from_tensor(data1);  // 길이 100
let dataset2 = TensorDataset::from_tensor(data2);  // 길이 50
let dataset3 = TensorDataset::from_tensor(data3);  // 길이 75

let combined = ConcatDataset::new(vec![dataset1, dataset2, dataset3]);

println!("Total length: {}", combined.len());  // 225

// combined.get(0~99): dataset1에서 가져옴
// combined.get(100~149): dataset2에서 가져옴
// combined.get(150~224): dataset3에서 가져옴
```

**특징:**
- 여러 데이터셋을 하나의 논리적 데이터셋으로 결합
- 내부적으로 cumulative sizes를 계산하여 효율적인 인덱싱
- 데이터를 복사하지 않고 참조만 유지

**사용 사례:**
- 여러 소스의 데이터 결합
- 데이터 증강 결과 병합
- 멀티 도메인 학습

### random_split

데이터셋을 학습/검증 세트로 무작위 분할합니다.

```rust
use hodu_utils::data::dataset::random_split;

let dataset = TensorDataset::from_tensors(data, labels);

// 80%는 학습용, 20%는 검증용
let (train_dataset, val_dataset) = random_split(dataset, 0.8, Some(42));

println!("Train size: {}", train_dataset.len());  // 80
println!("Val size: {}", val_dataset.len());      // 20
```

**파라미터:**
- `dataset`: 분할할 데이터셋 (Clone 구현 필요)
- `train_size`: 학습 데이터 비율 (0.0 ~ 1.0)
- `seed`: 재현성을 위한 시드값 (Option<u64>)

**특징:**
- Fisher-Yates shuffle 알고리즘 사용
- 데이터를 복사하지 않고 인덱스만 분할
- 시드값으로 재현 가능한 분할

**예제: 다양한 분할 비율:**

```rust
// 70% / 30% 분할
let (train, val) = random_split(dataset.clone(), 0.7, Some(42));

// 90% / 10% 분할
let (train, val) = random_split(dataset.clone(), 0.9, Some(42));

// 시드 없이 매번 다르게 분할
let (train, val) = random_split(dataset, 0.8, None);
```

## DataLoader

배치 단위로 데이터를 로딩하고 이터레이션합니다.

### 기본 사용법

```rust
use hodu_utils::data::dataloader::DataLoader;

let dataset = TensorDataset::from_tensors(data, labels);
let mut loader = DataLoader::new(dataset, 32);  // 배치 크기 32

// 배치 단위로 이터레이션
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

**기본 설정:**
- `batch_size`: 32
- `shuffle`: false
- `drop_last`: false
- `collate_fn`: default_collate

### DataLoader::builder()

더 많은 옵션을 설정할 수 있습니다.

```rust
let mut loader = DataLoader::builder(dataset)
    .batch_size(32)           // 배치 크기
    .shuffle(true)            // 에폭마다 셔플
    .drop_last(true)          // 마지막 불완전한 배치 제거
    .seed(42)                 // 재현성을 위한 시드값
    .build();

println!("Number of batches: {}", loader.num_batches());
```

**Builder 메서드:**

| 메서드 | 설명 | 기본값 |
|--------|------|--------|
| `batch_size(usize)` | 배치 크기 설정 | 1 |
| `shuffle(bool)` | 에폭마다 셔플 여부 | false |
| `drop_last(bool)` | 마지막 불완전한 배치 제거 | false |
| `seed(u64)` | 랜덤 시드 설정 | None |
| `collate_fn(CollateFn)` | 커스텀 collate 함수 | default_collate |

### 이터레이션 패턴

#### 패턴 1: while let 루프

```rust
let mut loader = DataLoader::new(dataset, 32);

while let Some(batch) = loader.next_batch()? {
    // 배치 처리
    if let Batch::Pair(data, labels) = batch {
        let predictions = model.forward(&data)?;
        let loss = criterion(&predictions, &labels)?;
    }
}
```

#### 패턴 2: Iterator 인터페이스

```rust
let mut loader = DataLoader::new(dataset, 32);

for batch_result in loader.iter_batches() {
    let batch = batch_result?;
    // 배치 처리
}
```

**Iterator의 장점:**
- Rust의 표준 Iterator 메서드 사용 가능
- `map`, `filter`, `collect` 등과 호환
- 함수형 프로그래밍 스타일

### 에폭 관리

```rust
let mut loader = DataLoader::builder(dataset)
    .batch_size(32)
    .shuffle(true)
    .seed(42)
    .build();

for epoch in 0..10 {
    println!("Epoch {}", epoch);

    // 이터레이션
    for batch_result in loader.iter_batches() {
        let batch = batch_result?;
        // 학습 로직
    }

    // 다음 에폭을 위해 리셋
    // shuffle=true인 경우 자동으로 재셔플
    loader.reset();
}
```

**reset() 동작:**
- 현재 인덱스를 0으로 초기화
- 에폭 카운터 증가
- `shuffle=true`이면 새로운 시드로 재셔플
  - 새로운 시드 = 원래 시드 + 에폭 번호

### 배치 크기와 drop_last

```rust
// 예: 데이터셋 크기가 100인 경우

// drop_last=false (기본값)
let loader1 = DataLoader::builder(dataset.clone())
    .batch_size(32)
    .drop_last(false)
    .build();
// 배치: [32, 32, 32, 4]  -> 총 4개 배치

// drop_last=true
let loader2 = DataLoader::builder(dataset)
    .batch_size(32)
    .drop_last(true)
    .build();
// 배치: [32, 32, 32]  -> 총 3개 배치 (마지막 4개 샘플 버려짐)
```

**drop_last를 true로 설정하는 경우:**
- Batch Normalization 사용 시 (배치 크기가 너무 작으면 불안정)
- 모든 배치가 동일한 크기여야 하는 경우
- 분산 학습에서 배치 크기를 일정하게 유지해야 하는 경우

### 커스텀 Collate 함수

배치 생성 로직을 커스터마이징할 수 있습니다.

```rust
use hodu_utils::data::batch::{Batch, DataItem};
use hodu_utils::data::dataloader::CollateFn;

// 커스텀 collate 함수 정의
fn my_collate(items: Vec<DataItem>) -> HoduResult<Batch> {
    // 예: 각 샘플을 정규화하면서 배치 생성
    let tensors: Vec<Tensor> = items
        .into_iter()
        .map(|item| match item {
            DataItem::Single(t) => {
                // 정규화: (x - mean) / std
                t.sub_scalar(0.5)?.div_scalar(0.5)
            }
            _ => panic!("Unexpected item type"),
        })
        .collect::<Result<Vec<_>, _>>()?;

    let refs: Vec<&Tensor> = tensors.iter().collect();
    let batched = Tensor::stack(&refs, 0)?;
    Ok(Batch::Single(batched))
}

// DataLoader에 적용
let loader = DataLoader::builder(dataset)
    .batch_size(32)
    .collate_fn(my_collate)
    .build();
```

**사용 사례:**
- 동적 패딩 (가변 길이 시퀀스)
- 데이터 증강
- 정규화
- 특수한 배치 포맷

## Sampler

데이터셋의 샘플링 순서를 제어합니다.

### SequentialSampler

순차적으로 샘플링합니다.

```rust
use hodu_utils::data::sampler::SequentialSampler;

let sampler = SequentialSampler::new(100);

// 0, 1, 2, ..., 99 순서로 이터레이션
for idx in sampler.iter() {
    println!("{}", idx);
}
```

**특징:**
- 가장 단순한 샘플러
- 인덱스가 0부터 length-1까지 순차적
- 재현 가능한 순서

### RandomSampler

무작위로 샘플링합니다.

```rust
use hodu_utils::data::sampler::RandomSampler;

// 100개 인덱스를 무작위로 셔플
let sampler = RandomSampler::new(100, Some(42));

for idx in sampler.iter() {
    println!("{}", idx);  // 무작위 순서: 73, 21, 5, ...
}
```

**특정 인덱스들만 셔플:**

```rust
let indices = vec![0, 5, 10, 15, 20];
let sampler = RandomSampler::with_indices(indices, Some(42));

for idx in sampler.iter() {
    println!("{}", idx);  // 5, 20, 0, 15, 10 등
}
```

**특징:**
- Fisher-Yates shuffle 알고리즘 사용
- 시드값으로 재현 가능
- 모든 인덱스를 정확히 한 번씩만 샘플링 (중복 없음)

### SubsetSampler

지정된 인덱스들만 샘플링합니다.

```rust
use hodu_utils::data::sampler::SubsetSampler;

let indices = vec![0, 2, 4, 6, 8];  // 짝수 인덱스만
let sampler = SubsetSampler::new(indices);

for idx in sampler.iter() {
    println!("{}", idx);  // 0, 2, 4, 6, 8
}
```

**사용 사례:**
- 특정 클래스만 샘플링
- 검증 세트 인덱스 지정
- 데이터 서브셋 정의

### BatchSampler

샘플러를 배치 단위로 그룹화합니다.

```rust
use hodu_utils::data::sampler::{SequentialSampler, BatchSampler};

let sampler = SequentialSampler::new(100);
let batch_sampler = BatchSampler::new(sampler, 32, false);

for batch_indices in batch_sampler.batches() {
    println!("Batch: {:?}", batch_indices);
    // [0, 1, ..., 31]
    // [32, 33, ..., 63]
    // [64, 65, ..., 95]
    // [96, 97, 98, 99]  (마지막 배치는 4개)
}
```

**drop_last 효과:**

```rust
// drop_last=false
let batch_sampler1 = BatchSampler::new(sampler.clone(), 32, false);
// 배치: [32, 32, 32, 4]

// drop_last=true
let batch_sampler2 = BatchSampler::new(sampler, 32, true);
// 배치: [32, 32, 32]  (마지막 4개 제거)
```

**주의사항:**
- BatchSampler는 인덱스의 배열의 배열을 반환
- DataLoader와는 별개로 사용 가능
- 커스텀 배치 로딩 로직 구현 시 유용

## Batch & DataItem

타입 안전한 데이터 표현을 위한 열거형입니다.

### DataItem

단일 샘플을 나타냅니다.

```rust
use hodu_utils::data::batch::DataItem;

// 단일 텐서 (예: 이미지만)
let item1 = DataItem::single(image_tensor);

// 데이터-레이블 쌍 (예: 이미지 + 클래스)
let item2 = DataItem::pair(image_tensor, label_tensor);

// 여러 텐서 (예: 이미지 + 세그멘테이션 마스크 + 바운딩 박스)
let item3 = DataItem::multiple(vec![image, mask, bbox]);
```

**열거형 정의:**

```rust
pub enum DataItem {
    Single(Tensor),
    Pair(Tensor, Tensor),
    Multiple(Vec<Tensor>),
}
```

### Batch

배치 데이터를 나타냅니다.

```rust
use hodu_utils::data::batch::Batch;

// 패턴 매칭으로 배치 처리
match batch {
    Batch::Single(tensor) => {
        // 단일 텐서 배치
        println!("Batch shape: {:?}", tensor.get_shape());  // [batch_size, ...]
    }
    Batch::Pair(data, labels) => {
        // 데이터-레이블 배치
        println!("Data shape: {:?}", data.get_shape());      // [batch_size, ...]
        println!("Labels shape: {:?}", labels.get_shape());  // [batch_size]
    }
    Batch::Multiple(tensors) => {
        // 여러 텐서 배치
        for (i, tensor) in tensors.iter().enumerate() {
            println!("Tensor {} shape: {:?}", i, tensor.get_shape());
        }
    }
}
```

**변환 메서드:**

```rust
// Batch::Pair를 튜플로 변환
if let Some((data, labels)) = batch.into_pair() {
    // data와 labels 사용
    let predictions = model.forward(&data)?;
}

// Batch::Single을 텐서로 변환
if let Some(tensor) = batch.into_single() {
    // tensor 사용
}

// Batch::Multiple을 Vec로 변환
if let Some(tensors) = batch.into_multiple() {
    // tensors 벡터 사용
}
```

**특징:**
- 타입 안전: 컴파일 타임에 배치 타입 체크
- 명시적: 어떤 데이터가 포함되어 있는지 명확
- 유연함: 다양한 데이터 구조 지원

### default_collate

기본 collate 함수의 동작을 이해합니다.

```rust
use hodu_utils::data::dataloader::default_collate;

// DataItem::Single들을 Batch::Single로 변환
let items = vec![
    DataItem::single(tensor1),  // [28, 28]
    DataItem::single(tensor2),  // [28, 28]
    DataItem::single(tensor3),  // [28, 28]
];
let batch = default_collate(items)?;
// Batch::Single([3, 28, 28])  <- stack(dim=0)으로 결합

// DataItem::Pair들을 Batch::Pair로 변환
let items = vec![
    DataItem::pair(data1, label1),
    DataItem::pair(data2, label2),
    DataItem::pair(data3, label3),
];
let batch = default_collate(items)?;
// Batch::Pair([3, ...], [3])  <- 각각 stack
```

**default_collate 동작:**
1. 모든 아이템이 같은 타입인지 확인
2. 각 텐서를 리스트로 수집
3. `Tensor::stack(&refs, 0)`로 dim=0에서 결합
4. 적절한 Batch 타입으로 반환

## Dataset Derive 매크로

`#[derive(Dataset)]`을 사용하여 커스텀 데이터셋을 쉽게 구현할 수 있습니다.

### 기본 사용법

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

    // Dataset trait을 위해 len()과 get() 구현
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

**매크로가 자동 생성하는 코드:**

```rust
impl Dataset for ImageDataset {
    fn len(&self) -> usize {
        self.len()  // 사용자가 정의한 len() 호출
    }

    fn get(&self, index: usize) -> HoduResult<DataItem> {
        self.get(index)  // 사용자가 정의한 get() 호출
    }
}
```

### 고급 예제: 이미지 증강 데이터셋

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

        // 데이터 증강 적용
        match item {
            DataItem::Pair(data, label) => {
                // 랜덤 플립
                let augmented = if rand::random::<bool>() {
                    data.flip(vec![2])?  // 가로 플립
                } else {
                    data
                };

                // 랜덤 노이즈 추가
                let noise = Tensor::randn_like(&augmented, 0.0, 0.1)?;
                let augmented = augmented.add(&noise)?;

                Ok(DataItem::pair(augmented, label))
            }
            _ => Ok(item),
        }
    }
}
```

## 실전 예제

### 예제 1: 기본 학습 루프

```rust
use hodu::prelude::*;
use hodu_utils::data::dataset::TensorDataset;
use hodu_utils::data::dataloader::DataLoader;
use hodu_utils::data::batch::Batch;

// 데이터 준비
let data = Tensor::randn(&[1000, 28, 28], DType::F32, Device::Cpu)?;
let labels = Tensor::randint(0, 10, &[1000], DType::I64, Device::Cpu)?;

// 데이터셋 및 DataLoader 생성
let dataset = TensorDataset::from_tensors(data, labels);
let mut loader = DataLoader::builder(dataset)
    .batch_size(32)
    .shuffle(true)
    .seed(42)
    .build();

// 학습 루프
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

### 예제 2: 학습/검증 분할

```rust
use hodu_utils::data::dataset::{TensorDataset, random_split};
use hodu_utils::data::dataloader::DataLoader;

// 데이터 준비
let data = Tensor::randn(&[1000, 784], DType::F32, Device::Cpu)?;
let labels = Tensor::randint(0, 10, &[1000], DType::I64, Device::Cpu)?;

// 데이터셋 생성 및 분할
let dataset = TensorDataset::from_tensors(data, labels);
let (train_dataset, val_dataset) = random_split(dataset, 0.8, Some(42));

println!("Train size: {}", train_dataset.len());  // 800
println!("Val size: {}", val_dataset.len());      // 200

// DataLoader 생성
let mut train_loader = DataLoader::builder(train_dataset)
    .batch_size(32)
    .shuffle(true)
    .drop_last(true)
    .seed(42)
    .build();

let mut val_loader = DataLoader::builder(val_dataset)
    .batch_size(32)
    .shuffle(false)  // 검증 세트는 셔플 안함
    .build();

// 학습 및 검증
for epoch in 0..10 {
    // 학습
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

    // 검증
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

### 예제 3: 커스텀 데이터셋과 증강

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
        // 파일에서 데이터 로드 (실제 구현 필요)
        let data = load_data_from_file(&self.data_paths[index])?;

        // Transform 적용
        let data = if self.transform {
            apply_augmentation(data)?
        } else {
            data
        };

        let label = Tensor::full(&[], self.labels[index] as f32)?;
        Ok(DataItem::pair(data, label))
    }
}

// 사용
let paths = vec![/* ... */];
let labels = vec![/* ... */];

let train_dataset = CustomDataset::new(paths.clone(), labels.clone(), true);
let val_dataset = CustomDataset::new(paths, labels, false);

let train_loader = DataLoader::new(train_dataset, 32);
let val_loader = DataLoader::new(val_dataset, 32);
```

### 예제 4: 멀티 소스 데이터 결합

```rust
use hodu_utils::data::dataset::{TensorDataset, ConcatDataset};

// 여러 소스의 데이터
let dataset1 = TensorDataset::from_tensors(data1, labels1);  // 소스 A
let dataset2 = TensorDataset::from_tensors(data2, labels2);  // 소스 B
let dataset3 = TensorDataset::from_tensors(data3, labels3);  // 소스 C

// 모든 데이터 결합
let combined = ConcatDataset::new(vec![dataset1, dataset2, dataset3]);

println!("Total samples: {}", combined.len());

// 결합된 데이터로 학습
let loader = DataLoader::builder(combined)
    .batch_size(64)
    .shuffle(true)
    .build();
```

### 예제 5: 불균형 데이터 처리

```rust
use hodu_utils::data::sampler::{RandomSampler, SubsetSampler};
use hodu_utils::data::dataset::Subset;

// 클래스별 인덱스 수집
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

// 소수 클래스 오버샘플링
let target_size = class_0_indices.len().max(class_1_indices.len());

// 클래스 0 서브셋 (그대로 사용)
let class_0_subset = Subset::new(dataset.clone(), class_0_indices);

// 클래스 1 서브셋 (반복 샘플링으로 크기 맞춤)
let mut balanced_class_1_indices = class_1_indices.clone();
while balanced_class_1_indices.len() < target_size {
    balanced_class_1_indices.extend(&class_1_indices);
}
balanced_class_1_indices.truncate(target_size);
let class_1_subset = Subset::new(dataset, balanced_class_1_indices);

// 균형 잡힌 데이터셋 생성
let balanced_dataset = ConcatDataset::new(vec![class_0_subset, class_1_subset]);

let loader = DataLoader::builder(balanced_dataset)
    .batch_size(32)
    .shuffle(true)
    .build();
```

## no_std 지원

`hodu_utils`는 `no_std` 환경을 지원합니다.

### Cargo.toml 설정

```toml
# no_std 환경 (기본 기능 비활성화)
[dependencies]
hodu_utils = { version = "0.1.9", default-features = false }

# std 환경 (표준 기능 활성화)
[dependencies]
hodu_utils = { version = "0.1.9", features = ["std"] }

# serde 지원 추가
[dependencies]
hodu_utils = { version = "0.1.9", features = ["std", "serde"] }
```

### 기능 플래그

| 플래그 | 설명 | 기본값 | 의존성 |
|--------|------|--------|--------|
| `std` | 표준 라이브러리 지원 | ✅ | - |
| `serde` | Serialization/Deserialization | ❌ | hodu_core/serde |

### no_std 사용 예제

```rust
#![no_std]

extern crate alloc;
use alloc::vec::Vec;

use hodu_utils::data::dataset::TensorDataset;
use hodu_utils::data::dataloader::DataLoader;

// no_std 환경에서도 동일하게 사용
let dataset = TensorDataset::from_tensor(data);
let loader = DataLoader::new(dataset, 32);
```

**제한사항:**
- `std::io` 관련 기능 사용 불가
- 일부 랜덤 생성 기능 제한적
- `alloc` crate 필요 (동적 메모리 할당)

## 주의사항

### 1. DataLoader의 reset() 호출

```rust
// 잘못된 사용: reset() 없이 재사용
let mut loader = DataLoader::new(dataset, 32);

// 첫 번째 에폭
for batch in loader.iter_batches() { /* ... */ }

// 두 번째 에폭 - 아무 배치도 반환되지 않음!
for batch in loader.iter_batches() { /* ... */ }  // ❌

// 올바른 사용: reset() 호출
let mut loader = DataLoader::new(dataset, 32);

for epoch in 0..10 {
    for batch in loader.iter_batches() { /* ... */ }
    loader.reset();  // ✅ 다음 에폭을 위해 리셋
}
```

### 2. drop_last와 Batch Normalization

```rust
// Batch Normalization 사용 시 drop_last=true 권장
let loader = DataLoader::builder(dataset)
    .batch_size(32)
    .drop_last(true)  // 마지막 작은 배치 제거
    .build();

// 이유: 배치 크기가 너무 작으면 통계량 계산이 불안정
// 예: 배치 크기 1이나 2는 분산 계산이 의미 없음
```

### 3. Shuffle 시드의 에폭별 변화

```rust
let mut loader = DataLoader::builder(dataset)
    .batch_size(32)
    .shuffle(true)
    .seed(42)
    .build();

// 에폭 0: 시드 42로 셔플
for batch in loader.iter_batches() { /* ... */ }
loader.reset();

// 에폭 1: 시드 43로 셔플 (42 + 1)
for batch in loader.iter_batches() { /* ... */ }
loader.reset();

// 에폭 2: 시드 44로 셔플 (42 + 2)
// ...
```

**재현성을 위해:**
- 동일한 시드값 사용
- 에폭마다 다른 셔플이 적용되지만 재현 가능

### 4. Clone 요구사항

```rust
// random_split은 Clone 필요
let dataset = TensorDataset::from_tensors(data, labels);
let (train, val) = random_split(dataset, 0.8, Some(42));  // ✅

// ConcatDataset은 소유권 이전
let dataset1 = TensorDataset::from_tensor(data1);
let dataset2 = TensorDataset::from_tensor(data2);
let combined = ConcatDataset::new(vec![dataset1, dataset2]);
// dataset1, dataset2는 이후 사용 불가

// 재사용하려면 미리 clone
let dataset1 = TensorDataset::from_tensor(data1);
let dataset1_clone = dataset1.clone();
let combined = ConcatDataset::new(vec![dataset1, dataset2]);
// dataset1_clone은 여전히 사용 가능
```

### 5. DataItem 타입 일관성

```rust
// default_collate은 모든 아이템이 같은 타입이어야 함

// ❌ 잘못된 사용
let items = vec![
    DataItem::single(tensor1),
    DataItem::pair(tensor2, label),  // 타입 불일치!
];
let batch = default_collate(items)?;  // Panic!

// ✅ 올바른 사용
let items = vec![
    DataItem::pair(data1, label1),
    DataItem::pair(data2, label2),
    DataItem::pair(data3, label3),
];
let batch = default_collate(items)?;  // OK
```

### 6. 인덱스 범위 체크

```rust
// Dataset::get()은 범위 체크를 수행해야 함

impl MyDataset {
    fn get(&self, index: usize) -> HoduResult<DataItem> {
        // ✅ 범위 체크
        if index >= self.len() {
            return Err(HoduError::InternalError(
                format!("Index {} out of bounds", index)
            ));
        }

        Ok(DataItem::single(self.data[index].clone()))
    }
}
```

## 함수 비교표

### Dataset 생성

| 함수 | 입력 | 출력 | 사용 사례 |
|------|------|------|----------|
| `TensorDataset::from_tensor()` | 단일 텐서 | DataItem::Single | 비지도 학습, 오토인코더 |
| `TensorDataset::from_tensors()` | 데이터, 레이블 | DataItem::Pair | 지도 학습, 분류/회귀 |
| `Subset::new()` | Dataset, 인덱스 | Subset | 데이터 분할, 서브셋 선택 |
| `ConcatDataset::new()` | Vec<Dataset> | ConcatDataset | 멀티 소스 결합 |
| `random_split()` | Dataset, 비율 | (Subset, Subset) | 학습/검증 분할 |

### DataLoader 설정

| 메서드 | 파라미터 | 효과 |
|--------|----------|------|
| `batch_size()` | usize | 배치 크기 설정 |
| `shuffle()` | bool | 에폭마다 셔플 여부 |
| `drop_last()` | bool | 마지막 불완전한 배치 제거 |
| `seed()` | u64 | 랜덤 시드 설정 |
| `collate_fn()` | CollateFn | 커스텀 collate 함수 |

### Sampler 종류

| Sampler | 순서 | 재현성 | 사용 사례 |
|---------|------|--------|----------|
| `SequentialSampler` | 순차적 (0, 1, 2, ...) | 항상 동일 | 평가, 디버깅 |
| `RandomSampler` | 무작위 (셔플) | 시드로 제어 | 학습, 데이터 증강 |
| `SubsetSampler` | 지정된 인덱스 순서 | 항상 동일 | 특정 샘플 선택 |
| `BatchSampler` | 배치 단위로 그룹화 | 기본 샘플러에 따름 | 커스텀 배치 로직 |

## 요약

| 카테고리 | 컴포넌트 | 주요 기능 |
|---------|----------|----------|
| **Dataset** | TensorDataset | 텐서로부터 데이터셋 생성 |
| | Subset | 데이터셋의 일부 선택 |
| | ConcatDataset | 여러 데이터셋 결합 |
| | random_split | 학습/검증 분할 |
| **DataLoader** | builder | 유연한 설정 |
| | iter_batches | Iterator 인터페이스 |
| | reset | 에폭 관리 |
| **Sampler** | Sequential | 순차적 샘플링 |
| | Random | 무작위 샘플링 |
| | Subset | 인덱스 기반 샘플링 |
| | Batch | 배치 그룹화 |
| **타입** | DataItem | 단일 샘플 표현 |
| | Batch | 배치 데이터 표현 |

**핵심 워크플로우:**
1. Dataset 생성 → 2. DataLoader 구성 → 3. 배치 이터레이션 → 4. 학습/검증

**Best Practices:**
- 학습에는 `shuffle=true`, 검증에는 `shuffle=false`
- Batch Normalization 사용 시 `drop_last=true`
- 에폭마다 `loader.reset()` 호출
- 재현성을 위해 시드값 설정
- 커스텀 데이터는 `#[derive(Dataset)]` 사용
