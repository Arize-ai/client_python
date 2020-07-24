from setuptools import setup, find_packages
from codecs import open

import sys, os


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path), encoding='utf-8') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


if sys.version_info < (3, 5, 3):
    sys.exit('Sorry, Python < 3.5.3 is not supported')

long_description = read('README.md')

# get the dependencies and installs
all_reqs = read('requirements.txt').splitlines()

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]
dependency_links = [
    x.strip().replace('git+', '') for x in all_reqs if x.startswith('git+')
]

test_reqs = read('requirements-dev.txt').splitlines()
test_requirements = [x.strip() for x in test_reqs if 'git+' not in x]

__version__ = get_version("arize/__init__.py")

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
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    keywords='arize',
    packages=find_packages(exclude=['docs', 'tests*']),
    include_package_data=True,
    install_requires=install_requires,
    tests_require=test_requirements,
    dependency_links=dependency_links,
)
