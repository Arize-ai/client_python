from setuptools import setup, find_packages
from codecs import open

import sys

__version__ = '0.0.8'

if sys.version_info < (3, 5, 3):
    sys.exit('Sorry, Python < 3.5.3 is not supported')

# Get the long description from the README file
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

# get the dependencies and installs
with open('requirements.txt', encoding='utf-8') as f:
    all_reqs = f.read().splitlines()

with open('requirements-dev.txt', encoding='utf-8') as f:
    all_reqs += f.read().splitlines()

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]
dependency_links = [
    x.strip().replace('git+', '') for x in all_reqs if x.startswith('git+')
]

setup(
    name='arize',
    version=__version__,
    description='A helper library to interact with Arize AI APIs',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Arize-ai/client_python',
    download_url='https://github.com/Arize-ai/client_python/tarball/' +
    __version__,
    author='Arize AI',
    author_email='support@arize.com',
    license='BSD',
    python_requires='>=3.5.3',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
    ],
    keywords='arize',
    packages=find_packages(exclude=['docs', 'tests*']),
    include_package_data=True,
    install_requires=install_requires,
    dependency_links=dependency_links,
)
