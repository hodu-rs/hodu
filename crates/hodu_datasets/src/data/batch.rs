use hodu_core::tensor::Tensor;

#[derive(Clone)]
pub enum DataItem {
    Single(Tensor),
    Pair(Tensor, Tensor),
    Multiple(Vec<Tensor>),
}

impl DataItem {
    pub fn single(tensor: Tensor) -> Self {
        DataItem::Single(tensor)
    }

    pub fn pair(data: Tensor, label: Tensor) -> Self {
        DataItem::Pair(data, label)
    }

    pub fn multiple(tensors: Vec<Tensor>) -> Self {
        DataItem::Multiple(tensors)
    }
}

#[derive(Clone)]
pub enum Batch {
    Single(Tensor),
    Pair(Tensor, Tensor),
    Multiple(Vec<Tensor>),
}

impl Batch {
    pub fn single(tensor: Tensor) -> Self {
        Batch::Single(tensor)
    }

    pub fn pair(data: Tensor, label: Tensor) -> Self {
        Batch::Pair(data, label)
    }

    pub fn multiple(tensors: Vec<Tensor>) -> Self {
        Batch::Multiple(tensors)
    }

    pub fn into_single(self) -> Option<Tensor> {
        match self {
            Batch::Single(t) => Some(t),
            _ => None,
        }
    }

    pub fn into_pair(self) -> Option<(Tensor, Tensor)> {
        match self {
            Batch::Pair(d, l) => Some((d, l)),
            _ => None,
        }
    }

    pub fn into_multiple(self) -> Option<Vec<Tensor>> {
        match self {
            Batch::Multiple(v) => Some(v),
            _ => None,
        }
    }
}
