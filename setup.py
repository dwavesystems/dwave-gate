# Confidential & Proprietary Information: D-Wave Systems Inc.
from setuptools import setup, find_packages

setup(
    name="dwgms",
    packages=find_packages(where="."),
    install_requires=[
        "numpy",
    ]
)