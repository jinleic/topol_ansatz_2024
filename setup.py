from setuptools import setup, find_packages

setup(
    name="topolansatz",
    version="0.1.0",
    packages=find_packages("src"),
    package_dir={'': 'src'},
    install_requires=[
        'qiskit>=1.0.0',
        'qiskit_aer',
        'networkx',
        'numpy',
        'pytest'
    ]
)
