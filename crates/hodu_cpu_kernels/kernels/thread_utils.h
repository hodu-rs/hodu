#ifndef THREAD_UTILS_H
#define THREAD_UTILS_H

#include <stddef.h>
#include <stdlib.h>

// Platform-specific headers for CPU detection
#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

// Platform-specific threading support
#if defined(ENABLE_THREADS)
#if defined(__unix__) || defined(__APPLE__) || (defined(_WIN32) && defined(__MINGW32__))
// POSIX threads (Unix, macOS, MinGW)
#define HAVE_PTHREAD
#include <pthread.h>
#elif defined(_WIN32)
// Windows native threads (MSVC, etc.)
#define HAVE_WIN32_THREADS
#include <process.h>
#endif
#endif

// Get number of CPU cores
static inline size_t get_num_threads(void) {
#if defined(ENABLE_THREADS)
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

// Thread abstraction layer
#if defined(HAVE_PTHREAD)
// POSIX threads
typedef pthread_t thread_t;

static inline int thread_create(thread_t *thread, void *(*start_routine)(void *), void *arg) {
    return pthread_create(thread, NULL, start_routine, arg);
}

static inline int thread_join(thread_t thread) { return pthread_join(thread, NULL); }

#elif defined(HAVE_WIN32_THREADS)
// Windows native threads
typedef HANDLE thread_t;

// Windows thread wrapper structure
typedef struct {
    void *(*start_routine)(void *);
    void *arg;
} win32_thread_args_t;

// Windows thread entry point wrapper
static inline unsigned __stdcall win32_thread_wrapper(void *arg) {
    win32_thread_args_t *wrapper_args = (win32_thread_args_t *)arg;
    void *(*start_routine)(void *) = wrapper_args->start_routine;
    void *thread_arg = wrapper_args->arg;
    free(wrapper_args);

    start_routine(thread_arg);
    return 0;
}

static inline int thread_create(thread_t *thread, void *(*start_routine)(void *), void *arg) {
    win32_thread_args_t *wrapper_args = (win32_thread_args_t *)malloc(sizeof(win32_thread_args_t));
    if (!wrapper_args) {
        return -1;
    }
    wrapper_args->start_routine = start_routine;
    wrapper_args->arg = arg;

    *thread = (HANDLE)_beginthreadex(NULL, 0, win32_thread_wrapper, wrapper_args, 0, NULL);
    if (*thread == 0) {
        free(wrapper_args);
        return -1;
    }
    return 0;
}

static inline int thread_join(thread_t thread) {
    WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
    return 0;
}

#else
// No threading support
typedef int thread_t;

static inline int thread_create(thread_t *thread, void *(*start_routine)(void *), void *arg) {
    (void)thread;
    (void)start_routine;
    (void)arg;
    return 0;
}

static inline int thread_join(thread_t thread) {
    (void)thread;
    return 0;
}
#endif

#endif // THREAD_UTILS_H
