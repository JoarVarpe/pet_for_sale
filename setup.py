from setuptools import setup, find_packages


setup(
    name = "for-sale-env",
    version = "0.0.1",
    author = "Joar Varpe",
    author_email = "joar.varpe@gmail.com",
    description = ("An environment made for playing steffan dorras for sale "),
    license = "BSD",
    keywords = "for sale env",
    url = "",
    packages=find_packages(
        where="src",
        include=["pkg*"],
        exclude=["additional"],

    ),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
)