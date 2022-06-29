#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import io
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import find_packages
from setuptools import setup

def read(*names, **kwargs):
    with io.open(
        join(dirname(__file__), *names),
        encoding=kwargs.get('encoding', 'utf8')
    ) as fh:
        return fh.read()


setup(
    name='daxbench',
    version='0.0.0',
    license='BSD-2-Clause',
    description="",
    long_description="",
    packages=find_packages(),
    package_dir={'': '.'},
    py_modules=[splitext(basename(path))[0] for path in glob('daxbench/*.py')],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: Unix',
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
        # uncomment if you test on these interpreters:
        # 'Programming Language :: Python :: Implementation :: IronPython',
        # 'Programming Language :: Python :: Implementation :: Jython',
        # 'Programming Language :: Python :: Implementation :: Stackless',
        'Topic :: Utilities',
        'Private :: Do Not Upload',
    ],
    keywords=[
        # eg: 'keyword1', 'keyword2', 'keyword3',
    ],
    python_requires='>=3.8',
    install_requires=[
        'typer',
        # 'jaxlib==0.3.14',
        # 'jax==0.3.14',
        'flax',
        'brax==0.0.13',
        'gym>=0.25,<0.26',
        'opencv-python==4.6.0.66',
        'imageio>=2.22,<2.23',
        'pyglet==1.5.26',
        'pyrender==0.1.45',
        'open3d',
        'polytope==0.2.3',
        'tensorflow==2.7.0',
        'tensorflow-addons==0.17.1',
        'tensorflow-hub==0.12.0',
        'meshcat',
        'pybullet',
        'usd-core',
        'transforms3d',
    ],
    extras_require={
        'dev': ['pytest']
        # eg:
        #   'rst': ['docutils>=0.11'],
        #   ':python_version=="2.6"': ['argparse'],
    },
    entry_points={
        'console_scripts': [
            'daxbench = daxbench.cli:main',
        ]
    },
)
