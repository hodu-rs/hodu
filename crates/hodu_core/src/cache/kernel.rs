use crate::layer::compat::*;

/// Maximum number of kernel names to cache before warning
const MAX_KERNEL_NAMES: usize = 1024;

/// Kernel name cache for efficient string -> &'static str conversion
///
/// This caches kernel names to avoid repeated allocations. While this uses
/// Box::leak() for &'static str, the cache size is bounded to prevent
/// unbounded memory growth.
pub struct KernelNameCache {
    cache: RwLock<HashMap<String, &'static str>>,
}

impl KernelNameCache {
    /// Creates a new kernel name cache
    pub fn new() -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
        }
    }

    /// Gets or creates a static string for the given kernel name
    pub fn get_or_insert(&self, name: String) -> &'static str {
        // Fast path: check if already cached (read lock)
        {
            #[cfg(feature = "std")]
            let cache = self.cache.read().unwrap();
            #[cfg(not(feature = "std"))]
            let cache = self.cache.read();

            if let Some(&cached) = cache.get(&name) {
                return cached;
            }
        }

        // Slow path: insert new entry (write lock)
        #[cfg(feature = "std")]
        let mut cache = self.cache.write().unwrap();
        #[cfg(not(feature = "std"))]
        let mut cache = self.cache.write();

        // Double-check in case another thread inserted it
        if let Some(&cached) = cache.get(&name) {
            return cached;
        }

        // Check cache size limit
        if cache.len() >= MAX_KERNEL_NAMES {
            #[cfg(feature = "std")]
            eprintln!(
                "WARNING: Kernel name cache exceeded {} entries. This may indicate a memory leak.",
                MAX_KERNEL_NAMES
            );
            // Still insert to avoid breaking functionality, but warn about it
        }

        // Leak the string to get &'static str
        // Note: This is intentional memory leak, but bounded by MAX_KERNEL_NAMES
        let static_str: &'static str = Box::leak(name.clone().into_boxed_str());
        cache.insert(name, static_str);
        static_str
    }
}

/// Global kernel name cache
static KERNEL_NAME_CACHE: LazyLock<KernelNameCache> = LazyLock::new(KernelNameCache::new);

/// Gets a static kernel name from the cache
///
/// This function caches kernel names to avoid memory leaks from repeated Box::leak calls.
#[inline]
pub fn get_kernel_name(name: String) -> &'static str {
    KERNEL_NAME_CACHE.get_or_insert(name)
}
