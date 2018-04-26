import os

from torch.utils.ffi import create_extension

this_file = os.path.dirname(os.path.realpath(__file__))

sources = ['src/conv_lib.c']
headers = ['include/conv_lib.h']
defines = [('WITH_CUDA', None)]

extra_objects = ['build/conv_cuda.o']
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]
ffi = create_extension(
    '_ext.inc_conv_lib',
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=True,
    extra_objects=extra_objects,
    extra_compile_args=["-std=c99"],
    include_dirs=[os.path.join(this_file, 'include')]
)

if __name__ == '__main__':
    ffi.build()
