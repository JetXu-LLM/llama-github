from setuptools import setup, find_packages

setup(
    name='llama_github',
    version='0.0.1',
    packages=find_packages(),
    package_data={
        'llama_github': ['config/config.json'],
    },
    include_package_data=True,
)