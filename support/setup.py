# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import glob
import os

import torch
from setuptools import setup
from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension

requirements = ["torch", "torchvision"]

'''

Install
$ python support/setup.py develop

部分编译信息：
    copying build/lib.linux-x86_64-3.7/support/_C.cpython-37m-x86_64-linux-gnu.so -> support
    Creating /home/hh/anaconda3/lib/python3.7/site-packages/support.egg-link (link to .)
    Adding support 0.1 to easy-install.pth file



Uninstall
$ python support/setup.py develop --uninstall

Test
$ python test/nms/test_nms.py
'''


'''/home/hh/anaconda3/lib/python3.7/site-packages目录下：
在support.egg-link有：
/home/hh/deeplearning_daily/easy-faster-rcnn.pytorch
.

在easy_install.pth中有：
/home/hh/deeplearning_daily/easy-faster-rcnn.pytorch

'''

'''##/home/hh/deeplearning_daily/easy-faster-rcnn.pytorch目录下：
support.egg-info子目录有文件：
    dependency_links.txt
    PKG-INFO
    SOURCES.txt
    top_level.txt->support

support子目录：
    _C.cpython-37m-x86_64-linux-gnu.so
这个文件与 /home/hh/anaconda3/lib/python3.7/site-packages/torch/_C.cpython-37m-x86_64-linux-gnu.so 同名，很奇葩.

'''

def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "src")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

    sources = main_file + source_cpu
    extension = CppExtension

    #extra_compile_args是个字典，这里先加入cxx
    extra_compile_args = {"cxx": []}
    define_macros = []

    if torch.cuda.is_available() and CUDA_HOME is not None:
        #如果cuda具备，把CppExtension替换为CUDAExtension
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        #这里再加入nvcc以及参数
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]

    #这一步没有必要，sources已经是绝对路径的文件了
    sources = [os.path.join(extensions_dir, s) for s in sources]

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "support._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


setup(
    name="support",
    version="0.1",
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
