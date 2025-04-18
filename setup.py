# setup.py
from setuptools import setup, find_packages

setup(
    name="gitcofl",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Add your dependencies here
        "GitPython",
        "schedule"
    ],
    author="Chokchai Faroongsarng",
    author_email="chokchai.fa@outlook.com",
    description="A Git-based Federated Learning Library",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Chokchai-Fa/GitCoFL",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)