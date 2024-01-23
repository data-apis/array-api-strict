from setuptools import setup, find_packages
import versioneer

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='array_api_strict',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(include=['array_api_strict*']),
    author="Consortium for Python Data API Standards",
    description="A strict, minimal implementation of the Python array API standard.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://data-apis.org/array-api-strict/",
    license="MIT",
    python_requires=">=3.9",
    install_requires=["numpy"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)
