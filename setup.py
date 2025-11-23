from setuptools import setup, find_packages

setup(
    name="ctg_viz",                          
    version="0.1.0",                         
    author="Josue Jimenez Apodaca",       
    description="Librería para Análisis Exploratorio de Datos CTG",
    long_description=open("readme.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/JosueJimenezA/Practica3_DCD.git", 
    packages=find_packages(),                
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "plotly",
        "streamlit",
        "pytest",
        "nbconvert"
    ],
)