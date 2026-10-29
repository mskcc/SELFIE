from setuptools import setup, find_packages
import os

# Utility function to read the README file.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

def read_requirements():
    with open('requirements.txt') as f:
        return f.read().splitlines()

setup(
    name='SELFIE',
    version='0.1.0',
    author='Melanie Schellenberg',
    author_email='schellem@mskcc.com',
    description='',
    license='MIT',
    keywords='',
    url='',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=read_requirements(),
    long_description=read('README.md'),
)
