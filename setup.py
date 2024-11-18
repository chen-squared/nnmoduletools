from setuptools import setup, find_packages

setup(
    name='nnmoduletools',
    version='0.1.2',
    description='A collection of neural network utilities',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
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