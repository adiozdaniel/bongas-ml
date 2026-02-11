"""
Setup script for BONGAS-ML package.

This script is kept for backward compatibility with older pip versions.
New installations should use pyproject.toml.
"""

from setuptools import setup, find_packages

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="bongas-ml",
    version="1.0.0",
    author="BONGAS Team",
    author_email="team@bongas.ai",
    description="Machine Learning package for BONGAS-AI recommendation system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Bongas-Squad/bongas-ml",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
            "pre-commit>=2.20.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinx-autodoc-typehints>=1.18.0",
        ],
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "bongas-ml=bongas_ml.__main__:main",
        ],
    },
    include_package_data=True,
    package_data={
        "bongas_ml": ["py.typed"],
    },
    zip_safe=False,
)