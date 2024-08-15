from setuptools import setup, find_packages
 
from pkg_resources import parse_requirements
with open("requirements.txt", encoding="utf-8") as fp:
    install_requires = [str(requirement) for requirement in parse_requirements(fp)]
 
setup(
    name="psyextract",
    version="1.0.0",
    author="RexZhann",
    author_email="zhanyunkai9@gmail.com",
    description="The package contains a framework for performing relation extraction using LLMs",
    long_description="eds sdk for python",


    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    
    
    include_package_data=True, # 一般不需要
    packages=find_packages(),
    install_requires=install_requires,
    entry_points={
        'console_scripts': [
            'test = test.help:main'
        ]
    }
)