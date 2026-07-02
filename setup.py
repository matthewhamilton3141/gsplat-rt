from setuptools import setup, find_packages
import os

kernels_dir = os.path.join(os.path.dirname(__file__), 'kernels')

try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    cuda_sources = sorted(
        os.path.join(kernels_dir, f)
        for f in os.listdir(kernels_dir)
        if f.endswith('.cu')
    ) if os.path.isdir(kernels_dir) else []

    ext_modules = []
    if cuda_sources:
        ext_modules.append(
            CUDAExtension(
                name='gaussian_kernels',
                sources=cuda_sources,
                extra_compile_args={
                    'cxx': ['-O3'],
                    'nvcc': [
                        '-O3',
                        '--use_fast_math',
                        '-arch=sm_86',  # Ampere (RTX 30xx/A-series); adjust for your GPU
                        '--ptxas-options=-v',
                    ],
                },
            )
        )

    cmdclass = {'build_ext': BuildExtension}

except ImportError:
    ext_modules = []
    cmdclass = {}

setup(
    name='gaussian_slam',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.10',
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
