# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='pyunmarked',
    version='0.1.0',
    description='A Python module for analyzing ecological data while accounting for imperfect detection',
    long_description=readme,
    long_description_content_type='text/markdown',
    python_requires=">=3.6",
    author='Ken Kellner',
    author_email='contact@kenkellner.com',
    url='https://github.com/kenkellner/pyunmarked',
    license=license,
    install_requires=[
        'numpy', 'pandas', 'scipy', 'patsy', 'prettytable'
    ],
    packages=find_packages(exclude=('tests', 'docs'))
)
