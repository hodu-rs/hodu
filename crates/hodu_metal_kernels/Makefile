# Directory settings
KERNEL_DIR = kernels
OUTPUT_DIR = build
INCLUDE_DIR = $(KERNEL_DIR)

# Output files
KERNEL_LIB = $(OUTPUT_DIR)/kernels.metallib

# Header files (not compiled directly)
HEADERS = $(KERNEL_DIR)/atomic.metal \
          $(KERNEL_DIR)/constants.metal \
          $(KERNEL_DIR)/utils.metal

# Source files
SOURCES = $(KERNEL_DIR)/ops_binary.metal \
		  $(KERNEL_DIR)/ops_concat_split.metal \
		  $(KERNEL_DIR)/ops_conv.metal \
		  $(KERNEL_DIR)/ops_indexing.metal \
		  $(KERNEL_DIR)/ops_matrix.metal \
		  $(KERNEL_DIR)/ops_reduce.metal \
          $(KERNEL_DIR)/ops_unary.metal

# Derived .air file paths
AIR_FILES = $(patsubst $(KERNEL_DIR)/%.metal,$(OUTPUT_DIR)/%.air,$(SOURCES))

# Compiler settings
METAL = xcrun -sdk macosx metal
METALLIB = xcrun -sdk macosx metallib
METAL_FLAGS = -I $(INCLUDE_DIR)

# Default target
all: prepare $(KERNEL_LIB)

# Create build directories
prepare:
	@mkdir -p $(OUTPUT_DIR)

# Compile source files to .air (depends on headers)
$(OUTPUT_DIR)/%.air: $(KERNEL_DIR)/%.metal $(HEADERS) | prepare
	$(METAL) $(METAL_FLAGS) -c $< -o $@

# Link all .air files into metallib
$(KERNEL_LIB): $(AIR_FILES) | prepare
	$(METALLIB) $^ -o $@

# Cleanup
clean:
	rm -rf $(OUTPUT_DIR)

# Dependency declarations
.PHONY: all prepare clean
