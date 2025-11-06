# Directory settings
KERNEL_DIR = kernels
OUTPUT_DIR = build
INCLUDE_DIR = $(KERNEL_DIR)
OPENBLAS_DIR = libs/OpenBLAS

# Output files
KERNEL_LIB = $(OUTPUT_DIR)/libhodu_cpu_kernels.a

# Header files (not compiled directly)
HEADERS = $(KERNEL_DIR)/atomic.h \
          $(KERNEL_DIR)/constants.h \
          $(KERNEL_DIR)/math_utils.h \
          $(KERNEL_DIR)/types.h \
          $(KERNEL_DIR)/utils.h \
          $(KERNEL_DIR)/ops_binary.h \
          $(KERNEL_DIR)/ops_concat_split.h \
          $(KERNEL_DIR)/ops_conv.h \
          $(KERNEL_DIR)/ops_indexing.h \
          $(KERNEL_DIR)/ops_matrix.h \
          $(KERNEL_DIR)/ops_reduce.h \
          $(KERNEL_DIR)/ops_unary.h \
          $(KERNEL_DIR)/ops_windowing.h

# Source files
SOURCES = $(KERNEL_DIR)/ops_binary.c \
          $(KERNEL_DIR)/ops_concat_split.c \
          $(KERNEL_DIR)/ops_conv.c \
          $(KERNEL_DIR)/ops_indexing.c \
          $(KERNEL_DIR)/ops_matrix.c \
          $(KERNEL_DIR)/ops_reduce.c \
          $(KERNEL_DIR)/ops_unary.c \
          $(KERNEL_DIR)/ops_windowing.c

# Derived .o file paths
OBJ_FILES = $(patsubst $(KERNEL_DIR)/%.c,$(OUTPUT_DIR)/%.o,$(SOURCES))

# Compiler settings
CC = clang
AR = ar
# Base flags
CFLAGS = -I $(INCLUDE_DIR) -std=c11 -Wall -Wextra -fPIC

# Optimization flags
OPT_FLAGS = -O3 -funroll-loops
# Auto-vectorization hints
OPT_FLAGS += -fvectorize -fslp-vectorize
# Safe fast math (preserve NaN/Inf)
OPT_FLAGS += -fno-math-errno -fno-trapping-math

# Platform-specific optimizations (use DISABLE_NATIVE=1 to disable)
ifndef DISABLE_NATIVE
    OPT_FLAGS += -march=native -mtune=native
endif

# SIMD feature detection (can be overridden)
ifndef DISABLE_SIMD
    # Compiler will use target-specific SIMD if available
    OPT_FLAGS += -DENABLE_SIMD_AUTO
endif

# Enable BLAS usage
OPT_FLAGS += -DUSE_BLAS

# Add OpenBLAS include path if it exists
ifneq ($(wildcard $(OPENBLAS_DIR)),)
    CFLAGS += -I$(OPENBLAS_DIR)
endif

CFLAGS += $(OPT_FLAGS)
ARFLAGS = rcs

# Default target
all: build_openblas prepare $(KERNEL_LIB)

# Build OpenBLAS if not already built
build_openblas:
	@if [ -d "$(OPENBLAS_DIR)" ] && [ ! -f "$(OPENBLAS_DIR)/libopenblas"*.a ]; then \
		echo "Building OpenBLAS..."; \
		ARCH=$$(uname -m); \
		OS=$$(uname -s); \
		case $$ARCH in \
			aarch64|arm64) TARGET=ARMV8 ;; \
			armv7*) TARGET=ARMV7 ;; \
			x86_64|amd64) TARGET=HASWELL ;; \
			i386|i686) TARGET=GENERIC ;; \
			riscv64) TARGET=RISCV64_GENERIC ;; \
			*) TARGET=GENERIC ;; \
		esac; \
		case $$OS in \
			MINGW*|MSYS*|CYGWIN*) MAKE_CMD=mingw32-make ;; \
			*) MAKE_CMD=make ;; \
		esac; \
		cd $(OPENBLAS_DIR) && $$MAKE_CMD TARGET=$$TARGET \
			NO_LAPACK=1 NO_FORTRAN=1 ONLY_CBLAS=1 NO_SHARED=1 \
			USE_THREAD=1 NUM_THREADS=64 libs -j$$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4); \
		echo "OpenBLAS built successfully for $$TARGET on $$OS"; \
	fi

# Create build directories
prepare:
	@mkdir -p $(OUTPUT_DIR)

# Compile source files to .o (depends on headers)
$(OUTPUT_DIR)/%.o: $(KERNEL_DIR)/%.c $(HEADERS) | prepare
	$(CC) $(CFLAGS) -c $< -o $@

# Create static library from all .o files
$(KERNEL_LIB): $(OBJ_FILES) | prepare
	$(AR) $(ARFLAGS) $@ $^

# Test compile (check for errors without creating library)
test: prepare
	@echo "Testing compilation..."
	@for src in $(SOURCES); do \
		echo "  Compiling $$src..."; \
		$(CC) $(CFLAGS) -fsyntax-only $$src || exit 1; \
	done
	@echo "All files compiled successfully!"

# Show compiler info
info:
	@echo "Compiler: $(CC)"
	@echo "Flags: $(CFLAGS)"
	@echo "Sources: $(SOURCES)"
	@echo "Headers: $(HEADERS)"
	@echo "Objects: $(OBJ_FILES)"
	@echo "Output: $(KERNEL_LIB)"

# Cleanup
clean:
	rm -rf $(OUTPUT_DIR)

# Clean OpenBLAS build files
clean_openblas:
	@if [ -d "$(OPENBLAS_DIR)" ]; then \
		echo "Cleaning OpenBLAS..."; \
		cd $(OPENBLAS_DIR) && make clean 2>/dev/null || true; \
	fi

# Dependency declarations
.PHONY: all build_openblas prepare test info clean clean_openblas
