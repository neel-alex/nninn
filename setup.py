from setuptools import find_packages, setup

setup(
    name="nninn",
    version="0.0.1",
    description="Experimenting with extracting information from neural network weights.",
    python_requires=">=3.7.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[],
)