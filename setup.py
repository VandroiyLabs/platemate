"""
"""

from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='PlateMate',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.0.1',

    description='Data analysis for 96 wells and multi-plate wells',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/VandroiyLabs/platemate',

    # Author details
    author='Thiago Mosqueiro',
    author_email='',

    license='MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Biotech academia and industry',
        'Topic :: Biotech :: Data analysis',

        'License :: MIT License',

        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
    ],

    keywords='plate reader, 96 wells',

    packages=find_packages(exclude=['contrib', 'docs', 'tests']),

    install_requires=['numpy','matplotlib'],


    extras_require={},
    package_data={},

    entry_points={
        'console_scripts': [
            'platemate=platemate:main',
        ],
    },
)
