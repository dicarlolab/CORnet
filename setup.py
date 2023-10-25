#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
    "torch>=0.4.0",
    "torchvision",
    "numpy",
    "pandas",
    "tqdm",
    "fire",
]

setup(
    name='CORnet',
    version='0.1.0',
    description="Modeling the Neural Mechanisms of Core Object Recognition ",
    long_description=readme,
    author="Jonas Kubilius, Martin Schrimpf",
    author_email='qbilius@mit.edu, msch@mit.edu',
    url='https://github.com/dicarlolab/CORnet',
    packages=['cornet'],
    include_package_data=True,
    install_requires=requirements,
    license="GNU GPL v3",
    zip_safe=False,
    keywords='CORnet Brain-Score',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU GPL v3',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6'
    ],
)
