from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='chamfer_3D_confidence',
    ext_modules=[
        CUDAExtension('chamfer_3D_confidence', [
            "/".join(__file__.split('/')[:-1] + ['chamfer_cuda_confidence.cpp']),
            "/".join(__file__.split('/')[:-1] + ['chamfer3D_confidence.cu']),
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
