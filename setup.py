from setuptools import find_packages, setup

from typing import List

REQUIREMENT_FILE_NAME = "requirements.txt"
HYPHEN_E_DOT = "-e ."

# provide all the libraries required for this project
# Create a function which can read the requirements.txt file and return all the library names to install_requires parameter
def get_requirements()->List[str]:    # function to return list of strings

    with open(REQUIREMENT_FILE_NAME) as requirement_file:
        requirement_list = requirement_file.readlines()
        # we have to replace new line char (\n) with nothing as it is not a part of library name
        requirement_list = [requirement_name.replace("\n","") for requirement_name in requirement_list]
        
        if HYPHEN_E_DOT in requirement_list:
            requirement_list.remove(HYPHEN_E_DOT)
        return requirement_list

setup(
    name = "sensor",
    version = "0.0.1",
    author = "irfan",
    author_email="mohdirfan57@gmail.com",
    packages=find_packages(),   # searches all folder where __init__.py file is present
    install_requires = get_requirements(),
)