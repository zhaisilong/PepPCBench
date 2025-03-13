from setuptools import setup, find_packages

setup(
    name="peppcbench",
    version="0.3.0",
    description="Comprehensive Benchmark for Protein-Peptide Complex Structure Prediction with All-Atom Protein Folding Neural Networks",
    author="Silong Zhai",
    author_email="zhaisilong@outlook.com",
    url="https://github.com/zhaisilong/PepPCBench",
    packages=["peppcbench"],
    install_requires=[],
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
