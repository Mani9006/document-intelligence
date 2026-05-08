"""
Setup script for the Automated Document Intelligence System.

Uses setuptools for package installation. The project is primarily
configured via pyproject.toml; this file provides backward compatibility.
"""

from setuptools import find_packages, setup

setup(
    name="doc-intelligence",
    version="1.0.0",
    description=(
        "Automated Document Intelligence System - "
        "PDF text extraction, classification, entity extraction, "
        "and similarity analysis"
    ),
    author="Mani",
    author_email="myfamily9006@gmail.com",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "PyPDF2>=3.0.0",
        "pdfplumber>=0.10.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "docintel=src.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: General",
    ],
)
