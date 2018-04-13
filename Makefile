# Unix commands.
PYTHON := python
RM_RF := rm -rf


# File structure.
BUILD_DIR := build
INCLUDE_DIRS := include
TORCH_FFI_BUILD := build_ffi.py
TEST := test.py
CONV_CUDA := $(BUILD_DIR)/conv_cuda.o
TORCH_FFI_TARGET := $(BUILD_DIR)/conv_lib.o

INCLUDE_FLAGS := $(foreach d, $(INCLUDE_DIRS), -I$d)

all: $(TORCH_FFI_TARGET)

$(TORCH_FFI_TARGET): $(CONV_CUDA) $(TORCH_FFI_BUILD)
	$(PYTHON) $(TORCH_FFI_BUILD)
	$(PYTHON) $(TEST)

$(BUILD_DIR)/conv_cuda.o: src/conv_cuda.cu
	@ mkdir -p $(BUILD_DIR)
	nvcc -lcublas_device -lcudadevrt src/conv_cuda.cu -o conv_cuda.o -Xcompiler -fPIC -rdc=true -shared -Iinclude -arch=sm_60
	@ mv *.o $(BUILD_DIR)/

clean:
	$(RM_RF) $(BUILD_DIR) $(CONV_CUDA)
