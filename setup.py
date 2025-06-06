#!/usr/bin/env python3
"""
Setup script for UOR/Prime Axioms Factorizer
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(req_path):
        with open(req_path, "r", encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    return []

setup(
    name="factorizer",
    version="1.0.0",
    author="UOR Research",
    author_email="research@uor.dev",
    description="Mathematical factorization using Universal Organizing Resonance (UOR) principles",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/uor-research/factorizer",
    packages=find_packages(exclude=["test", "test.*", "*.test", "*.test.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: Public Domain",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.900",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "factorize=factorizer.cli:main",
        ],
    },
    keywords="factorization mathematics prime numbers fibonacci golden ratio spectral analysis",
    project_urls={
        "Bug Reports": "https://github.com/uor-research/factorizer/issues",
        "Source": "https://github.com/uor-research/factorizer",
        "Documentation": "https://factorizer.readthedocs.io/",
    },
)