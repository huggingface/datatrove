from setuptools import find_packages, setup


REQUIRED_PKGS = [
    "cchardet==2.1.7",
    "inscriptis==2.3.2",
    "loguru==0.7.0",
    "multiprocess==0.70.14",
    "nltk==3.8.1",
    "numpy==1.25.0",
    "python-magic==0.4.27",
    "readability-lxml @ git+https://github.com/huggingface/python-readability.git@speedup",
    "trafilatura==1.6.1",
    "warcio==1.7.4",
    "zstandard==0.21.0",
    "pyarrow==12.0.1",
    "tokenizers==0.13.3",
    "tldextract==3.4.4",
    "pandas==2.0.3",
    "backoff==2.2.1",
    "build==1.0.0",
]

EXTRAS = {
    "dev": [
        "black~=23.1",
        "pre-commit>=3.3.3",
        "pytest>=7.2.0",
        "pytest-timeout",
        "pytest-xdist",
        "ruff>=0.0.241,<=0.0.259",
    ]
}

setup(
    name="datatrove",
    version="0.0.1.dev0",  # expected format is one of x.y.z.dev0, or x.y.z.rc1 or x.y.z (no to dashes, yes to dots)
    description="HuggingFace library to process and filter large amounts of webdata",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="HuggingFace Inc.",
    author_email="guilherme@huggingface.co",
    url="https://github.com/huggingface/datatrove",
    license="Apache 2.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={"": ["assets/*"]},
    python_requires=">=3.7.0",
    install_requires=REQUIRED_PKGS,
    extras_require=EXTRAS,
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="data machine learning processing",
    entry_points={
        "console_scripts": [
            "check_dataset=datatrove.tools.check_dataset:main",
            "merge_stats=datatrove.tools.merge_stats:main",
        ]
    },
)
