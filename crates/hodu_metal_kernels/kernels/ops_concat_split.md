# Metal Concat/Split Operations Usage Guide

## CONCATENATION OPERATIONS

### Overview
The concat operation combines multiple input tensors along a specified dimension into a single output tensor. All input tensors must be passed as a single combined buffer.

### Usage Example

Concatenating three tensors along dimension 1:
- **Input 0**: shape [2, 3, 4] (24 elements)
- **Input 1**: shape [2, 5, 4] (40 elements)
- **Input 2**: shape [2, 2, 4] (16 elements)
- **Concat dimension**: 1
- **Output**: shape [2, 10, 4] (80 elements)

### Step 1: Combine Input Tensors

```cpp
// Combine all input tensors into a single buffer
std::vector<float> combined_input;
std::vector<size_t> buffer_offsets;

buffer_offsets.push_back(0);
combined_input.insert(combined_input.end(), input0.begin(), input0.end());

buffer_offsets.push_back(combined_input.size());  // 24
combined_input.insert(combined_input.end(), input1.begin(), input1.end());

buffer_offsets.push_back(combined_input.size());  // 64
combined_input.insert(combined_input.end(), input2.begin(), input2.end());
```

### Step 2: Prepare Metadata

```cpp
size_t num_dims = 3;
size_t num_inputs = 3;
size_t concat_dim = 1;

std::vector<size_t> metadata;

// output_shape: [2, 10, 4]
metadata.push_back(2);
metadata.push_back(10);
metadata.push_back(4);

// concat_dim
metadata.push_back(concat_dim);  // 1

// num_inputs
metadata.push_back(num_inputs);  // 3

// input_shapes: each input's shape (num_inputs * num_dims = 9 elements)
metadata.push_back(2); metadata.push_back(3); metadata.push_back(4);  // input0
metadata.push_back(2); metadata.push_back(5); metadata.push_back(4);  // input1
metadata.push_back(2); metadata.push_back(2); metadata.push_back(4);  // input2

// input_strides: each input's strides (num_inputs * num_dims = 9 elements)
metadata.push_back(12); metadata.push_back(4); metadata.push_back(1);  // input0: [3*4, 4, 1]
metadata.push_back(20); metadata.push_back(4); metadata.push_back(1);  // input1: [5*4, 4, 1]
metadata.push_back(8);  metadata.push_back(4); metadata.push_back(1);  // input2: [2*4, 4, 1]

// input_offsets: each input's offset (num_inputs = 3 elements, typically 0)
metadata.push_back(0);
metadata.push_back(0);
metadata.push_back(0);

// input_buffer_offsets: starting position of each input in combined buffer (num_inputs = 3)
metadata.push_back(0);   // input0 starts at position 0
metadata.push_back(24);  // input1 starts at position 24
metadata.push_back(64);  // input2 starts at position 64
```

### Metadata Layout

```
Index     Content
[0..2]    output_shape = [2, 10, 4]
[3]       concat_dim = 1
[4]       num_inputs = 3
[5..13]   input_shapes = [2,3,4, 2,5,4, 2,2,4]
[14..22]  input_strides = [12,4,1, 20,4,1, 8,4,1]
[23..25]  input_offsets = [0, 0, 0]
[26..28]  input_buffer_offsets = [0, 24, 64]

Total metadata size: 29 elements
```

### Step 3: Execute Metal Kernel

```objc
size_t num_els = 2 * 10 * 4;  // Output total elements = 80

id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

[encoder setComputePipelineState:pipelineState];
[encoder setBuffer:inputBuffer offset:0 atIndex:0];      // combined_input
[encoder setBuffer:outputBuffer offset:0 atIndex:1];     // output
[encoder setBytes:&num_els length:sizeof(size_t) atIndex:2];
[encoder setBytes:&num_dims length:sizeof(size_t) atIndex:3];
[encoder setBuffer:metadataBuffer offset:0 atIndex:4];   // metadata

MTLSize gridSize = MTLSizeMake(num_els, 1, 1);
MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
[encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];

[encoder endEncoding];
[commandBuffer commit];
```

---

## SPLIT OPERATIONS

### Overview
The split operation splits a single input tensor into multiple output tensors along a specified dimension. Each output requires a separate kernel invocation.

### Usage Example

Splitting one tensor along dimension 1:
- **Input**: shape [2, 10, 4] (80 elements)
- **Split dimension**: 1
- **Output 0**: shape [2, 3, 4] (24 elements) - takes indices 0-2
- **Output 1**: shape [2, 5, 4] (40 elements) - takes indices 3-7
- **Output 2**: shape [2, 2, 4] (16 elements) - takes indices 8-9

### Step 1: Prepare Metadata for Each Output

For **Output 0**:
```cpp
size_t num_dims = 3;
size_t split_dim = 1;

std::vector<size_t> metadata;

// input_shape: [2, 10, 4]
metadata.push_back(2);
metadata.push_back(10);
metadata.push_back(4);

// input_strides: [40, 4, 1]  (computed as [10*4, 4, 1])
metadata.push_back(40);
metadata.push_back(4);
metadata.push_back(1);

// input_offset: 0 (typically)
metadata.push_back(0);

// split_dim: 1
metadata.push_back(split_dim);

// output_size_on_dim: 3 (size of this output along split dimension)
metadata.push_back(3);

// split_offset: 0 (starting index in input along split dimension)
metadata.push_back(0);
```

For **Output 1**:
```cpp
// ... same as above until split_offset

// output_size_on_dim: 5
metadata.push_back(5);

// split_offset: 3 (starts at index 3 in the input)
metadata.push_back(3);
```

For **Output 2**:
```cpp
// ... same as above until split_offset

// output_size_on_dim: 2
metadata.push_back(2);

// split_offset: 8 (starts at index 8 in the input)
metadata.push_back(8);
```

### Metadata Layout

```
Index         Content
[0..num_dims-1]       input_shape
[num_dims..2*num_dims-1]  input_strides
[2*num_dims]          input_offset
[2*num_dims+1]        split_dim
[2*num_dims+2]        output_size_on_dim (for this specific output)
[2*num_dims+3]        split_offset (where this output starts in input)

For num_dims=3: Total metadata size = 10 elements
```

### Step 2: Execute Metal Kernel (for each output)

For **Output 0**:
```objc
size_t num_els = 2 * 3 * 4;  // 24 elements

id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

[encoder setComputePipelineState:pipelineState];
[encoder setBuffer:inputBuffer offset:0 atIndex:0];       // input tensor
[encoder setBuffer:output0Buffer offset:0 atIndex:1];     // output 0
[encoder setBytes:&num_els length:sizeof(size_t) atIndex:2];
[encoder setBytes:&num_dims length:sizeof(size_t) atIndex:3];
[encoder setBuffer:metadata0Buffer offset:0 atIndex:4];   // metadata for output 0

MTLSize gridSize = MTLSizeMake(num_els, 1, 1);
MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
[encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];

[encoder endEncoding];
[commandBuffer commit];
```

Repeat similar calls for Output 1 and Output 2 with their respective metadata and buffer sizes.
