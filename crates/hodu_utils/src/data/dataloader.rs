use crate::compat::*;
use hodu_core::{error::HoduResult, tensor::Tensor};

use super::batch::{Batch, DataItem};
use super::dataset::Dataset;

pub type CollateFn = fn(Vec<DataItem>) -> HoduResult<Batch>;

pub fn default_collate(items: Vec<DataItem>) -> HoduResult<Batch> {
    if items.is_empty() {
        return Err(hodu_core::error::HoduError::InvalidArgument(
            "Cannot collate empty items".into(),
        ));
    }

    match &items[0] {
        DataItem::Single(_) => {
            let tensors: Vec<Tensor> = items
                .into_iter()
                .map(|item| match item {
                    DataItem::Single(t) => t,
                    _ => panic!("Inconsistent data types in batch"),
                })
                .collect();

            let refs: Vec<&Tensor> = tensors.iter().collect();
            let batched = Tensor::stack(&refs, 0)?;
            Ok(Batch::Single(batched))
        },
        DataItem::Pair(_, _) => {
            let mut data_tensors = Vec::new();
            let mut label_tensors = Vec::new();

            for item in items {
                match item {
                    DataItem::Pair(d, l) => {
                        data_tensors.push(d);
                        label_tensors.push(l);
                    },
                    _ => panic!("Inconsistent data types in batch"),
                }
            }

            let data_refs: Vec<&Tensor> = data_tensors.iter().collect();
            let label_refs: Vec<&Tensor> = label_tensors.iter().collect();

            let batched_data = Tensor::stack(&data_refs, 0)?;
            let batched_labels = Tensor::stack(&label_refs, 0)?;

            Ok(Batch::Pair(batched_data, batched_labels))
        },
        DataItem::Multiple(vec) => {
            let num_tensors = vec.len();
            let mut tensor_vecs: Vec<Vec<Tensor>> = vec![Vec::new(); num_tensors];

            for item in items {
                match item {
                    DataItem::Multiple(tensors) => {
                        if tensors.len() != num_tensors {
                            panic!("Inconsistent number of tensors in batch");
                        }
                        for (i, tensor) in tensors.into_iter().enumerate() {
                            tensor_vecs[i].push(tensor);
                        }
                    },
                    _ => panic!("Inconsistent data types in batch"),
                }
            }

            let mut batched = Vec::with_capacity(num_tensors);
            for tensor_vec in tensor_vecs {
                let refs: Vec<&Tensor> = tensor_vec.iter().collect();
                batched.push(Tensor::stack(&refs, 0)?);
            }

            Ok(Batch::Multiple(batched))
        },
    }
}

pub struct DataLoaderBuilder<D: Dataset> {
    dataset: D,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
    collate_fn: CollateFn,
}

impl<D: Dataset> DataLoaderBuilder<D> {
    pub fn new(dataset: D) -> Self {
        Self {
            dataset,
            batch_size: 1,
            shuffle: false,
            drop_last: false,
            collate_fn: default_collate,
        }
    }

    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    pub fn drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }

    pub fn collate_fn(mut self, collate_fn: CollateFn) -> Self {
        self.collate_fn = collate_fn;
        self
    }

    pub fn build(self) -> DataLoader<D> {
        DataLoader::new_with_config(
            self.dataset,
            self.batch_size,
            self.shuffle,
            self.drop_last,
            self.collate_fn,
        )
    }
}

pub struct DataLoader<D: Dataset> {
    dataset: D,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
    collate_fn: CollateFn,
    indices: Vec<usize>,
    current_idx: usize,
}

impl<D: Dataset> DataLoader<D> {
    pub fn new(dataset: D, batch_size: usize) -> Self {
        Self::new_with_config(dataset, batch_size, false, false, default_collate)
    }

    pub fn new_with_config(
        dataset: D,
        batch_size: usize,
        shuffle: bool,
        drop_last: bool,
        collate_fn: CollateFn,
    ) -> Self {
        let len = dataset.len();
        let mut indices: Vec<usize> = (0..len).collect();

        if shuffle {
            // TODO: Implement proper shuffling when we have RNG support
            // For now, just use the sequential indices
        }

        Self {
            dataset,
            batch_size,
            shuffle,
            drop_last,
            collate_fn,
            indices,
            current_idx: 0,
        }
    }

    pub fn builder(dataset: D) -> DataLoaderBuilder<D> {
        DataLoaderBuilder::new(dataset)
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn num_batches(&self) -> usize {
        let len = self.dataset.len();
        if self.drop_last {
            len / self.batch_size
        } else {
            (len + self.batch_size - 1) / self.batch_size
        }
    }

    pub fn reset(&mut self) {
        self.current_idx = 0;
        if self.shuffle {
            // TODO: Implement proper shuffling
        }
    }

    pub fn has_next(&self) -> bool {
        if self.current_idx >= self.dataset.len() {
            return false;
        }

        if self.drop_last {
            let remaining = self.dataset.len() - self.current_idx;
            remaining >= self.batch_size
        } else {
            true
        }
    }

    pub fn next_batch(&mut self) -> HoduResult<Option<Batch>> {
        if !self.has_next() {
            return Ok(None);
        }

        let start_idx = self.current_idx;
        let end_idx = (start_idx + self.batch_size).min(self.dataset.len());

        let batch_indices = &self.indices[start_idx..end_idx];
        let mut items = Vec::with_capacity(batch_indices.len());

        for &idx in batch_indices {
            items.push(self.dataset.get(idx)?);
        }

        self.current_idx = end_idx;

        let batch = (self.collate_fn)(items)?;
        Ok(Some(batch))
    }

    pub fn iter_batches(&mut self) -> DataLoaderIterator<D> {
        DataLoaderIterator { loader: self }
    }
}

pub struct DataLoaderIterator<'a, D: Dataset> {
    loader: &'a mut DataLoader<D>,
}

impl<'a, D: Dataset> Iterator for DataLoaderIterator<'a, D> {
    type Item = HoduResult<Batch>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.loader.next_batch() {
            Ok(Some(batch)) => Some(Ok(batch)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}
