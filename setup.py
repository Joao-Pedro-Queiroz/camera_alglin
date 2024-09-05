from setuptools import setup, find_packages


setup(
    name="enigma",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[],
    author="João Pedro Queiroz Viana, Pedro Fardin",
    author_email="joaopqv@al.insper.edu.br, ",
    description="Uma biblioteca de criptografia.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Joao-Pedro-Queiroz/enigma",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)