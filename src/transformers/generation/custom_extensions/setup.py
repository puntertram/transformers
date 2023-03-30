from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='logits_process_cuda',
    ext_modules=[
        CUDAExtension('logits_process_cuda', [
            'logits_process.cpp',
            'logits_process_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
