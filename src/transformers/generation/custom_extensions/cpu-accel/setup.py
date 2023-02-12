from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cpu_accel',
    ext_modules=[
        CUDAExtension('cpu_accel', [
            'cpu_accel.cpp'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
