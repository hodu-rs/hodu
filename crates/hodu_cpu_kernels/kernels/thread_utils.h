#ifndef THREAD_UTILS_H
#define THREAD_UTILS_H

#include <stddef.h>

// Enable multi-threading for std builds with pthread (unless explicitly disabled)
#if defined(ENABLE_THREADS) &&                                                                     \
    (defined(__unix__) || defined(__APPLE__) || (defined(_WIN32) && defined(__MINGW32__)))
#define HAVE_PTHREAD
#include <pthread.h>

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

// Get number of CPU cores
static inline size_t get_num_threads(void) {
#if defined(__unix__) || defined(__APPLE__)
    long nproc = sysconf(_SC_NPROCESSORS_ONLN);
    return (nproc > 0) ? (size_t)nproc : 1;
#elif defined(_WIN32)
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    return (size_t)sysinfo.dwNumberOfProcessors;
#else
    return 1;
#endif
}

// Determine optimal number of threads for workload
// Returns 1 for small workloads to avoid threading overhead
static inline size_t get_optimal_threads(size_t workload_size, size_t min_work_per_thread) {
    if (workload_size < min_work_per_thread * 2) {
        return 1; // Too small for parallelization
    }

    size_t max_threads = get_num_threads();
    size_t optimal_threads = workload_size / min_work_per_thread;

    return (optimal_threads < max_threads) ? optimal_threads : max_threads;
}

#else
// No pthread support
static inline size_t get_num_threads(void) { return 1; }

static inline size_t get_optimal_threads(size_t workload_size, size_t min_work_per_thread) {
    (void)workload_size;
    (void)min_work_per_thread;
    return 1;
}
#endif

#endif // THREAD_UTILS_H
