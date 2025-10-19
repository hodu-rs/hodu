# Directory settings
KERNEL_DIR = kernels
OUTPUT_DIR = build
INCLUDE_DIR = $(KERNEL_DIR)

# Output files
KERNEL_LIB = $(OUTPUT_DIR)/kernels.metallib

# Utils file
UTILS = $(KERNEL_DIR)/utils.metal
UTILS_AIR = $(OUTPUT_DIR)/utils.air

# Source files
SOURCES = $(KERNEL_DIR)/binary.metal \
          $(KERNEL_DIR)/unary.metal

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

# Compile utility files
$(UTILS_AIR): $(UTILS) | prepare
	$(METAL) $(METAL_FLAGS) -c $< -o $@

# Compile source files to .air
$(OUTPUT_DIR)/%.air: $(KERNEL_DIR)/%.metal $(UTILS_AIR) | prepare
	$(METAL) $(METAL_FLAGS) -c $< -o $@

# Link all .air files into metallib
$(KERNEL_LIB): $(UTILS_AIR) $(AIR_FILES) | prepare
	$(METALLIB) $^ -o $@

# Cleanup
clean:
	rm -rf $(OUTPUT_DIR)

# Dependency declarations
.PHONY: all prepare clean
