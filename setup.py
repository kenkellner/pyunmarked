# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

#with open('README.rst') as f:
#    readme = f.read()

#with open('LICENSE') as f:
#    license = f.read()

setup(
    name='pyunmarked',
    version='0.1.0',
    description='',
    long_description='',
    python_requires=">=3.6",
    author='Ken Kellner',
    author_email='contact@kenkellner.com',
    url='',
    license='',
    install_requires=[
        'numpy', 'pandas', 'scipy', 'patsy', 'prettytable'
    ],
    packages=find_packages(exclude=('tests', 'docs'))
)
