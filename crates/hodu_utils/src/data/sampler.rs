use crate::compat::*;

pub trait Sampler {
    fn iter(&self) -> Box<dyn Iterator<Item = usize>>;

    fn len(&self) -> usize;
}

pub struct SequentialSampler {
    length: usize,
}

impl SequentialSampler {
    pub fn new(length: usize) -> Self {
        Self { length }
    }
}

impl Sampler for SequentialSampler {
    fn iter(&self) -> Box<dyn Iterator<Item = usize>> {
        Box::new(0..self.length)
    }

    fn len(&self) -> usize {
        self.length
    }
}

pub struct RandomSampler {
    indices: Vec<usize>,
}

impl RandomSampler {
    pub fn new(length: usize) -> Self {
        // TODO: Implement proper shuffling when we have RNG support
        let indices: Vec<usize> = (0..length).collect();
        Self { indices }
    }

    pub fn with_indices(indices: Vec<usize>) -> Self {
        // TODO: Shuffle the indices
        Self { indices }
    }
}

impl Sampler for RandomSampler {
    fn iter(&self) -> Box<dyn Iterator<Item = usize>> {
        Box::new(self.indices.clone().into_iter())
    }

    fn len(&self) -> usize {
        self.indices.len()
    }
}

pub struct SubsetSampler {
    indices: Vec<usize>,
}

impl SubsetSampler {
    pub fn new(indices: Vec<usize>) -> Self {
        Self { indices }
    }
}

impl Sampler for SubsetSampler {
    fn iter(&self) -> Box<dyn Iterator<Item = usize>> {
        Box::new(self.indices.clone().into_iter())
    }

    fn len(&self) -> usize {
        self.indices.len()
    }
}

pub struct BatchSampler<S: Sampler> {
    sampler: S,
    batch_size: usize,
    drop_last: bool,
}

impl<S: Sampler> BatchSampler<S> {
    pub fn new(sampler: S, batch_size: usize, drop_last: bool) -> Self {
        Self {
            sampler,
            batch_size,
            drop_last,
        }
    }

    pub fn batches(&self) -> Vec<Vec<usize>> {
        let indices: Vec<usize> = self.sampler.iter().collect();
        let mut batches = Vec::new();

        for chunk in indices.chunks(self.batch_size) {
            if self.drop_last && chunk.len() < self.batch_size {
                break;
            }
            batches.push(chunk.to_vec());
        }

        batches
    }
}
