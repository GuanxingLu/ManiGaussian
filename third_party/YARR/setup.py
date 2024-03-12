import codecs
import os

import setuptools


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


def get_install_requires():
    install_requires = []
    with open('requirements.txt') as f:
        for req in f:
            install_requires.append(req.strip())
    return install_requires


setuptools.setup(
    version=get_version("yarr/__init__.py"),
    name='yarr',
    author='Stephen James',
    author_email='slj12@ic.ac.uk',
    packages=setuptools.find_packages(),
    install_requires=get_install_requires()
)
