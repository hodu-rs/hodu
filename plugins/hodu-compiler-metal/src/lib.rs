//! Metal Compiler Plugin for Hodu
//!
//! Compiles Hodu Snapshots to Metal artifacts (.metallib)
//!
//! ## Compilation Flow
//!
//! ```text
//! Snapshot (IR)
//!     │
//!     ▼
//! ┌─────────────────────┐
//! │ Generate Dispatch   │  Snapshot → DispatchPlan
//! │ Manifest            │
//! └─────────────────────┘
//!     │
//!     ▼
//! ┌─────────────────────┐
//! │ Bundle Kernels      │  hodu_metal_kernels → .metal
//! └─────────────────────┘
//!     │
//!     ▼
//! ┌─────────────────────┐
//! │ xcrun metal         │  .metal → .air
//! └─────────────────────┘
//!     │
//!     ▼
//! ┌─────────────────────┐
//! │ xcrun metallib      │  .air → .metallib
//! └─────────────────────┘
//!     │
//!     ▼
//! CompiledArtifact (metallib + dispatch manifest)
//! ```

mod compiler;
mod dispatch;
mod xcrun;

pub use compiler::MetalCompiler;
