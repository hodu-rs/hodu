//! CPU Compiler Plugin for Hodu
//!
//! Compiles Hodu Snapshots to native shared libraries (.so/.dylib/.dll)
//!
//! ## Compilation Flow
//!
//! ```text
//! Snapshot (IR)
//!     │
//!     ▼
//! ┌─────────────────────┐
//! │ Generate Dispatch   │  Snapshot → DispatchManifest
//! │ Manifest            │
//! └─────────────────────┘
//!     │
//!     ▼
//! ┌─────────────────────┐
//! │ Generate C Code     │  Manifest → main.c (kernel calls)
//! └─────────────────────┘
//!     │
//!     ▼
//! ┌─────────────────────┐
//! │ C Compiler          │  main.c + libhodu_cpu_kernels.a → .so/.dylib/.dll
//! │ (clang/gcc/cl.exe)  │
//! └─────────────────────┘
//!     │
//!     ▼
//! CompiledArtifact (shared library + dispatch manifest)
//! ```
//!
//! ## Platform Support
//!
//! | Platform | Default Compiler | Alternatives |
//! |----------|------------------|--------------|
//! | macOS    | clang            | gcc          |
//! | Linux    | gcc              | clang        |
//! | Windows  | cl.exe (MSVC)    | gcc (MinGW)  |

mod codegen;
mod compiler;
mod dispatch;
mod toolchain;

pub use compiler::CpuCompiler;
