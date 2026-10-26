from setuptools import setup, find_packages

setup(
    name="multimodal-explainable-diagnosis",
    version="1.0.0",
    author="Akhand Pratap Singh Chandel",
    author_email="akhandpratapsingh857@gmail.com",
    description="Multimodal Deep Learning for Explainable Medical Diagnosis with Image and Clinical Fusion, Missing Data Robustness, and Grad-CAM/SHAP XAI",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "tensorflow>=2.15.0",
        "keras>=3.0.0",
        "numpy>=1.24.0,<2.0.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pillow>=9.5.0",
        "opencv-python>=4.8.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "python-multipart>=0.0.6",
        "requests>=2.31.0",
        "tqdm>=4.65.0",
        "pydantic>=2.0.0",
    ],
)
