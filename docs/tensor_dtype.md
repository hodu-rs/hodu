# DType

| Data type | DType | Not supported |
|-----------|-------|---------------|
| boolean                   | `hodu::bool`                                |            |
| 8-bit floating point      | `hodu::f8e4m3`                              | metal, xla |
| 8-bit floating point      | `hodu::f8e5m2`                              | metal, xla |
| 16-bit floating point     | `hodu::bfloat16`, `hodu::bf16`              |            |
| 16-bit floating point     | `hodu::float16`, `hodu::f16`, `hodu::half`  |            |
| 32-bit floating point     | `hodu::float32`, `hodu::f32`                |            |
| 64-bit floating point     | `hodu::float64`, `hodu::f64`                | metal      |
| 8-bit integer (unsigned)  | `hodu::uint8`, `hodu::u8`                   |            |
| 16-bit integer (unsigned) | `hodu::uint16`, `hodu::u16`                 |            |
| 32-bit integer (unsigned) | `hodu::uint32`, `hodu::u32`                 |            |
| 64-bit integer (unsigned) | `hodu::uint64`, `hodu::u64`                 | metal      |
| 8-bit integer (signed)    | `hodu::int8`, `hodu::i8`                    |            |
| 16-bit integer (signed)   | `hodu::int16`, `hodu::i16`                  |            |
| 32-bit integer (signed)   | `hodu::int32`, `hodu::i32`                  |            |
| 64-bit integer (signed)   | `hodu::int64`, `hodu::i64`                  | metal      |
