from setuptools import setup, find_packages

setup(
    name='nnmoduletools',
    version='0.0.2-alpha',
    description='A collection of neural network utilities',
    author='chensquared',
    author_email='chensquared319@gmail.com',
    packages=find_packages(),
    install_package_data=True,
    include_package_data=True,
    install_requires=[
        'numpy',
        'torch',
        'matplotlib',
        'tqdm'
    ],
    classifiers=[
        "Operating System :: OS Independent",
        'Programming Language :: Python :: 3.10',
    ],
    python_requires=">=3.10",
)