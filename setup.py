from setuptools import setup, find_packages


def parse_requirements(filename):
    """Load requirements from a file."""
    with open(filename, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


setup(
    name="scalable_gp_inference",
    version="0.1.0",
    author="Pratik Rathore",
    author_email="pratikr@stanford.edu",
    description="A proof-of-concept for scalable Gaussian Process inference",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/pratikrathore8/scalable_gp_inference",
    license="Apache-2.0",
    packages=find_packages(
        include=["scalable_gp_inference", "scalable_gp_inference.*"]
    ),
    package_dir={"scalable_gp_inference": "scalable_gp_inference"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=parse_requirements("requirements.txt")
    + [
        "rlaopt @ git+https://github.com/udellgroup/rlaopt.git@689902c7af366fa99d04434cbb639e63073c0bf8"  # noqa: E501
    ],
)
