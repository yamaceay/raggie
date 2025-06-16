from setuptools import setup, find_packages

requires = [
    "accelerate==1.7.0",
    "datasets==3.6.0",
    "faiss-cpu==1.11.0",
    "matplotlib==3.9.2",
    "numpy==2.3.0",
    "scikit-learn==1.5.2",
    "scipy==1.15.3",
    "seaborn==0.13.2",
    "sentence-transformers==4.1.0",
    "torch==2.5.1"
]

setup(
    name="raggie",
    version="0.1.4",
    url='https://yamaceay.github.io/raggie/raggie.html',
    author="Yamac Eren Ay",
    author_email="yamacerenay2001@gmail.com",
    description="A Python package for training, retrieval, and visualization of key-value embeddings.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    zip_safe=False,
    platforms='any',
    python_requires=">=3.6",
    install_requires=requires,
)