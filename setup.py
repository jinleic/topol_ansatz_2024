# setup.py
from setuptools import setup, find_packages

setup(
    name="topolansatz",
    version="0.1",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "qiskit",
        "qiskit-aer",
        "networkx",
        "scipy",
        "numpy",
        "matplotlib",
    ]
)