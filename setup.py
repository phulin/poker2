from setuptools import setup, find_packages

setup(
    name="alphaholdem",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
        "pyyaml>=5.4.0",
    ],
    python_requires=">=3.8",
)
