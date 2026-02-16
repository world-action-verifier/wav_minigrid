from setuptools import setup, find_packages

setup(
    name="asim_minigrid",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
