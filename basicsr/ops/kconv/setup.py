import os
import torch

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

cxx_args = ['-std=c++11']

nvcc_args = [
    '-gencode', 'arch=compute_50,code=sm_50',
    '-gencode', 'arch=compute_52,code=sm_52',
    '-gencode', 'arch=compute_60,code=sm_60',
    '-gencode', 'arch=compute_61,code=sm_61'
]


def make_cuda_ext(name, module, sources, sources_cuda=[]):

    define_macros = []
    extra_compile_args = {'cxx': cxx_args, 'nvcc': nvcc_args}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        sources += sources_cuda
    else:
        print(f'Compiling {name} without CUDA')
        extension = CppExtension

    # print([os.path.join('', p) for p in sources])
    # import pdb; pdb.set_trace()
    return extension(
        name=f'{module}.{name}',
        sources=[os.path.join('', p) for p in sources],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)


setup(
    version='1.0.0',
    name='kernel_conv_ext',
    ext_modules=[
        make_cuda_ext(
            name='kernel_conv_ext',
            module='basicsr.models.ops.kconv',
            sources=[],
            sources_cuda=[
                'src/KernelConv2D_cuda.cpp',
                'src/KernelConv2D_kernel.cu'
            ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
