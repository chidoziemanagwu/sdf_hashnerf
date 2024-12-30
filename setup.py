from setuptools import setup, find_packages

setup(
    name="sdf_hashnerf",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'torch>=1.9.0',
        'numpy>=1.19.2',
        'matplotlib>=3.3.4',
        'tqdm>=4.61.0',
        'opencv-python>=4.5.3',
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="SDF implementation in HashNeRF",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.7",
)