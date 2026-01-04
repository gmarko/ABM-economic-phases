from setuptools import setup, find_packages

setup(
    name="abm-economic-phases",
    version="0.1.0",
    author="Marco Durán Cabobianco",
    author_email="marco@anachroni.co",
    description="Agent-Based Model for Economic Phases with External Variables",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/mduran/ABM-economic-phases",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "networkx>=2.6.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
