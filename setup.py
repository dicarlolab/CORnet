#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
    "torch>=0.4.1+",
    "torchvision",
    "numpy",
    "pandas",
    "tqdm",
    "fire",
]

setup(
    name='CORnet',
    version='0.1.0',
    description="A cortex-like neural network for object recognition",
    long_description=readme,
    author="Jonas Kubilius, Martin Schrimpf",
    author_email='qbilius@mit.edu, msch@mit.edu',
    url='https://github.com/dicarlolab/CORnet',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='CORnet brain-score',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU GPL v3',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
