from setuptools import setup, find_packages

setup(
    name="aesthtify",
    version="1.0.0",
    description="Aesthetic scoring and layout analysis toolkit for interior spaces",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(include=["utils", "routes", "interior_analysis", "utils.*", "routes.*"]),
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
            'aesthtify=app:main',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)