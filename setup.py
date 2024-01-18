from setuptools import find_packages, setup


install_requires = [
    "huggingface-hub>=0.17.0",
    "boto3>=1.28.78",
    "faust-cchardet>=2.1.19",
    "inscriptis>=2.3.2",
    "loguru>=0.7.0",
    "multiprocess>=0.70.14",
    "nltk>=3.8.1",
    "numpy>=1.25.0",
    "python-magic>=0.4.27",
    "readability-lxml @ git+https://github.com/huggingface/python-readability.git@speedup",
    "trafilatura>=1.6.1",
    "warcio>=1.7.4",
    "pyarrow>=12.0.1",
    "tokenizers>=0.13.3",
    "tldextract>=3.4.4",
    "pandas>=2.0.3",
    "backoff>=2.2.1",
    "fsspec>=2023.9.2",
    "humanize>=4.8.0",
    "rich>=13.7.0",
]

extras = {}

extras["quality"] = [
    "ruff>=0.1.5",
]

extras["tests"] = [
    "pytest",
    "pytest-timeout",
    "pytest-xdist",
    # Optional dependencies
    "fasttext-wheel",
    "moto[s3,server]",
    "s3fs",
]

extras["all"] = extras["quality"] + extras["tests"]

extras["dev"] = extras["all"]

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
    include_package_data=True,
    python_requires=">=3.10.0",
    install_requires=install_requires,
    extras_require=extras,
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="data machine learning processing",
    entry_points={
        "console_scripts": [
            "check_dataset=datatrove.tools.check_dataset:main",
            "merge_stats=datatrove.tools.merge_stats:main",
            "launch_pickled_pipeline=datatrove.tools.launch_pickled_pipeline:main",
            "failed_logs=datatrove.tools.failed_logs:main",
            "inspect_data=datatrove.tools.inspect_data:main",
        ]
    },
)
