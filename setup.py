import sys

from setuptools import setup, find_packages

version = "0.0.3"

if len(sys.argv) >= 3 and sys.argv[1] == "validate_tag":
    if sys.argv[2] != version:
        raise Exception(
            f"A versão TAG [{sys.argv[2]}] é diferente da versão no arquivo setup.py [{version}]."
        )
    exit()


setup(
    **{
        "name": "simplifiedbert",
        "description": "SimplifiedBert is a Python package that simplifies the training and evaluation process for BERT models.",
        "long_description": open("README.md").read(),
        "long_description_content_type": "text/markdown",
        "license": "Apache-2.0",
        "author": "Raphael Silva Fontes",
        "author_email": "raphael.fontes@lais.huol.ufrn.br",
        "packages": find_packages(),
        "version": version,
        "download_url": f"https://github.com/lais-huol/SimplifiedBERT/releases/tag/{version}",
        "url": "https://github.com/lais-huol/SimplifiedBERT",
        "keywords": [
            "BERT",
            "Simplify",
            "NLP",
            "Evaluation",
            "Train",
        ],
        "python_requires": ">=3.10",
        "install_requires": [
            "scikit-learn",
            "pandas",
            "transformers",
            "torch",
            "transformers[torch]",
            "accelerate",
            
        ],
        "classifiers": [
            "Development Status :: 2 - Pre-Alpha",
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
        ],
    }
)