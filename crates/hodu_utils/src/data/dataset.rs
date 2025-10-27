use crate::{compat::*, data::batch::DataItem};
use hodu_core::{error::HoduResult, tensor::Tensor};
pub use hodu_utils_macros::Dataset;

pub trait Dataset {
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn get(&self, index: usize) -> HoduResult<DataItem>;
}

pub struct TensorDataset {
    data: Tensor,
    labels: Option<Tensor>,
}

impl TensorDataset {
    pub fn from_tensor(data: Tensor) -> Self {
        Self { data, labels: None }
    }

    pub fn from_tensors(data: Tensor, labels: Tensor) -> Self {
        Self {
            data,
            labels: Some(labels),
        }
    }
}

impl Dataset for TensorDataset {
    fn len(&self) -> usize {
        self.data.get_shape()[0]
    }

    fn get(&self, index: usize) -> HoduResult<DataItem> {
        let data_item = self.data.slice(0, index, index + 1, 1)?.squeeze(&[0])?;

        if let Some(ref labels) = self.labels {
            let label_item = labels.slice(0, index, index + 1, 1)?.squeeze(&[0])?;
            Ok(DataItem::Pair(data_item, label_item))
        } else {
            Ok(DataItem::Single(data_item))
        }
    }
}

pub struct Subset<D: Dataset> {
    dataset: D,
    indices: Vec<usize>,
}

impl<D: Dataset> Subset<D> {
    pub fn new(dataset: D, indices: Vec<usize>) -> Self {
        Self { dataset, indices }
    }
}

impl<D: Dataset> Dataset for Subset<D> {
    fn len(&self) -> usize {
        self.indices.len()
    }

    fn get(&self, index: usize) -> HoduResult<DataItem> {
        if index >= self.indices.len() {
            return Err(hodu_core::error::HoduError::InvalidArgument(format!(
                "Index {} out of bounds for subset of length {}",
                index,
                self.indices.len()
            )));
        }
        self.dataset.get(self.indices[index])
    }
}

pub struct ConcatDataset<D: Dataset> {
    datasets: Vec<D>,
    cumulative_sizes: Vec<usize>,
}

impl<D: Dataset> ConcatDataset<D> {
    pub fn new(datasets: Vec<D>) -> Self {
        let mut cumulative_sizes = Vec::with_capacity(datasets.len());
        let mut total = 0;
        for dataset in &datasets {
            total += dataset.len();
            cumulative_sizes.push(total);
        }
        Self {
            datasets,
            cumulative_sizes,
        }
    }
}

impl<D: Dataset> Dataset for ConcatDataset<D> {
    fn len(&self) -> usize {
        self.cumulative_sizes.last().copied().unwrap_or(0)
    }

    fn get(&self, index: usize) -> HoduResult<DataItem> {
        if index >= self.len() {
            return Err(hodu_core::error::HoduError::InvalidArgument(format!(
                "Index {} out of bounds for dataset of length {}",
                index,
                self.len()
            )));
        }

        for (i, &cumulative_size) in self.cumulative_sizes.iter().enumerate() {
            if index < cumulative_size {
                let dataset_idx = if i == 0 {
                    index
                } else {
                    index - self.cumulative_sizes[i - 1]
                };
                return self.datasets[i].get(dataset_idx);
            }
        }

        unreachable!()
    }
}

pub fn random_split<D: Dataset>(dataset: D, train_size: f32) -> (Subset<D>, Subset<D>)
where
    D: Clone,
{
    let total_len = dataset.len();
    let train_len = (total_len as f32 * train_size) as usize;

    // TODO: Implement proper shuffling when we have RNG support
    let mut indices: Vec<usize> = (0..total_len).collect();

    let train_indices = indices[..train_len].to_vec();
    let val_indices = indices[train_len..].to_vec();

    let train_dataset = Subset::new(dataset.clone(), train_indices);
    let val_dataset = Subset::new(dataset, val_indices);

    (train_dataset, val_dataset)
}
