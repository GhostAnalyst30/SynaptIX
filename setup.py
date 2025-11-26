from setuptools import setup, find_packages

# Cargar README como descripción larga
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "SynaptixX – Librería unificada de IA, ML, DL, Visión y NLP para Python."

setup(
    name="synaptix",  # ← más único para PyPI
    version="0.1.0",
    author="Emmanuel Ascendra Perez",
    author_email="ascendraemmanuel@gmail.com",
    description="Librería de Inteligencia Artificial para Python (ML, DL, Visión, NLP, IA Clásica).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Immanuel3008/SynaptIX",
    project_urls={
        "Repositorio": "https://github.com/Immanuel3008/SynaptIX",
        "Reportar Bugs": "https://github.com/Immanuel3008/SynaptIX/issues",
        "Documentación": "https://github.com/Immanuel3008/SynaptIX/wiki",
    },
    packages=find_packages(),
    include_package_data=True,

    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],

    python_requires=">=3.8",

    # Dependencias base
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
    ],

    # Dependencias opcionales
    extras_require={
        "ml": [
            "scikit-learn>=1.0.0",
            "statsmodels>=0.13.0",
        ],
        "dl": [
            "tensorflow>=2.5.0",
        ],
        "vision": [
            "opencv-python>=4.5.0",
        ],
        "nlp": [
            "nltk>=3.6.0",
            "scikit-learn>=1.0.0",  # TF-IDF
        ],
        "viz": [
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ],
        "all": [
            "scikit-learn>=1.0.0",
            "statsmodels>=0.13.0",
            "tensorflow>=2.5.0",
            "opencv-python>=4.5.0",
            "nltk>=3.6.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ],
    },

    keywords=[
        "machine learning",
        "deep learning",
        "artificial intelligence",
        "data science",
        "computer vision",
        "nlp",
        "synaptixx",
    ],
)
