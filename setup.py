from setuptools import find_packages, setup

setup(
    name="wav_minigrid",
    version="0.1",
    packages=find_packages(where="src") + ["env"],
    package_dir={
        "": "src",
        "env": "env",
    },
)
