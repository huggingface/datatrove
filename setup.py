from setuptools import find_packages, setup


REQUIRED_PKGS = [
    "warcio==1.7.4",
    "cchardet==2.1.7",
    "python-magic==0.4.27",
    "multiprocess==0.70.14",
    "numpy==1.25.0",
    "nltk==3.8.1",
    "loguru==0.7.0"
]

setup(
    name="datatrove",
    version="0.0.1",  # expected format is one of x.y.z.dev0, or x.y.z.rc1 or x.y.z (no to dashes, yes to dots)
    description="HuggingFace library to process and filter large amounts of webdata",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="HuggingFace Inc.",
    author_email="guilherme@huggingface.co",
    url="https://github.com/huggingface/datatrove",
    license="Apache 2.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.7.0",
    install_requires=REQUIRED_PKGS,
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
    keywords="data machine learning processing"
)
