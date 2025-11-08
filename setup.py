from setuptools import setup, find_packages

setup(
    name="loaf",
    version="0.1.0",
    description="LOAF - Market Analysis System with ML Capabilities",
    author="benjicoriat",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "yfinance",
        "stable-baselines3",
        "pandas",
        "numpy",
        "matplotlib",
        "requests",
        "beautifulsoup4",
        "transformers",
        "torch",
        "tqdm",
    ],
    python_requires=">=3.8",
)