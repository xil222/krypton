import os
from torch.utils.ffi import create_extension
import glob

this_file = os.path.dirname(os.path.realpath(__file__))

sources = ['src/conv_lib.c']
headers = ['include/conv_lib.h']
defines = [('WITH_CUDA', None)]

extra_objects = ['build/conv_cuda.o']
# extra_objects = ['build/a.out']#['build/conv_cuda.o']#, 'build/conv_cuda.dc.o']
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]
# extra_objects.append('/usr/local/cuda/lib64/libcudart.so')
# extra_objects.append('/usr/local/cuda/lib64/libcublas.so')
# extra_objects.append('/usr/local/cuda/lib64/libcudadevrt.a')
# extra_objects.append('/usr/local/cuda/lib64/libcublas_device.a')
# extra_link_args = []
# extra_link_args += glob.glob('/usr/local/cuda/lib64/*.a')

ffi = create_extension(
    '_ext.inc_conv_lib',
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=True,
    extra_objects=extra_objects,
    extra_compile_args=["-std=c99"],
    #extra_link_args=['-L/usr/local/cuda/lib64 -lcudart -lcublas -lcudadevrt -lcublas_device'],
    include_dirs=[os.path.join(this_file, 'include')]
)

if __name__ == '__main__':
    ffi.build()
