#need to be in main file : pip install -e .

from setuptools import setup

package_name = "librairies"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    install_requires=["setuptools"],
    zip_safe=True,
    license="TODO",
)