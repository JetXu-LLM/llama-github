from setuptools import setup, find_packages

# Read the requirements from requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.readlines()

setup(
    install_requires=[req.strip() for req in requirements],
)