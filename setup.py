from setuptools import find_packages, setup


install_requires = [
    "dill>=0.3.0",
    "fsspec>=2023.6.0",
    "huggingface-hub>=0.17.0",
    "humanize",
    "loguru>=0.7.0",
    "multiprocess",
    "numpy>=1.25.0",
    "tqdm",
]

extras = {}

extras["cli"] = [
    "rich",
]

extras["io"] = [
    "faust-cchardet",
    "pyarrow",
    "python-magic",
    "warcio",
]

extras["s3"] = [
    "s3fs>=2023.12.2",
]

extras["processing"] = [
    "fasttext-wheel",
    "nltk",
    "inscriptis",
    "readability-lxml @ git+https://github.com/huggingface/python-readability.git@speedup",
    "tldextract",
    "trafilatura",
    "tokenizers",
]

extras["quality"] = [
    "ruff>=0.1.5",
]

extras["testing"] = (
    extras["cli"]
    + extras["io"]
    + extras["processing"]
    + extras["s3"]
    + [
        "pytest",
        "pytest-timeout",
        "pytest-xdist",
        "moto[s3,server]",
    ]
)

extras["all"] = extras["quality"] + extras["testing"]

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
