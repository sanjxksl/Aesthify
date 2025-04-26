"""
===============================================================================
Setup Script for Project Packaging and Distribution
===============================================================================
Manages project metadata, dependencies, and distribution packaging.
"""

from setuptools import setup, find_packages

setup(
    name="aesthify",  # Project name
    version="1.0.0",   # Version
    description="Aesthetic scoring and layout analysis toolkit for interior spaces",
    author="KSL Sanjana",
    author_email="sanjxksl@gmail.com",
    packages=find_packages(include=[
        "utils", 
        "routes", 
        "interior_analysis", 
        "utils.*", 
        "routes.*"
    ]),
    include_package_data=True,
    install_requires=[
        "flask",
        "numpy",
        "opencv-python",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "ultralytics",
        "python-dotenv",
        "openpyxl"
    ],
    entry_points={
        'console_scripts': [
            'aesthify=app:main',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
