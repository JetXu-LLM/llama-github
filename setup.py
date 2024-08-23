from setuptools import setup, find_packages
from configparser import ConfigParser

# Read the requirements from requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.readlines()

# Read version from setup.cfg
config = ConfigParser()
config.read('setup.cfg')
version = config['metadata']['version']

setup(
    version=version,
    install_requires=[req.strip() for req in requirements],
)