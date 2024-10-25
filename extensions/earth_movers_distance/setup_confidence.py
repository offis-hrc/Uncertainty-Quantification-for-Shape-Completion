from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='emd_cuda_confidence',
    ext_modules=[
        CUDAExtension(
            name='emd_cuda_confidence',
            sources=[
                'emd_confidence.cpp',
                'emd_kernel_confidence.cu',
            ],
            # extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']}
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
 