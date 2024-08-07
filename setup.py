import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="marbleregression", # Replace with your own username
    version="0.0.1",
    author="Manuel Baum",
    author_email="manuelbaum@posteo.de",
    description="code for estimating the number of marbles in a tray-like end-effector",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/manuelbaum/marbleregression",
    project_urls={
        "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=["torch"]
)
