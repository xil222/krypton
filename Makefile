# Unix commands.
PYTHON := python
NVCC_COMPILE := nvcc -c -o
RM_RF := rm -rf

# Library compilation rules.
NVCC_FLAGS := -x cu -Xcompiler -fPIC -shared -lcublas -gencode arch=compute_35,code=sm_35 -rdc=true -lcudadevrt -lcublas_device

# File structure.
BUILD_DIR := build
INCLUDE_DIRS := include
TORCH_FFI_BUILD := build_ffi.py
CONV_CUDA := $(BUILD_DIR)/conv_cuda.so
TORCH_FFI_TARGET := $(BUILD_DIR)/conv_lib.so

INCLUDE_FLAGS := $(foreach d, $(INCLUDE_DIRS), -I$d)

all: $(TORCH_FFI_TARGET)

$(TORCH_FFI_TARGET): $(CONV_CUDA) $(TORCH_FFI_BUILD)
	$(PYTHON) $(TORCH_FFI_BUILD)

$(BUILD_DIR)/%.so: src/%.cu
	@ mkdir -p $(BUILD_DIR)
	# Separate cpp shared library that will be loaded to the extern C ffi
	$(NVCC_COMPILE) $@ $? $(NVCC_FLAGS) $(INCLUDE_FLAGS)

clean:
	$(RM_RF) $(BUILD_DIR) $(CONV_CUDA)
