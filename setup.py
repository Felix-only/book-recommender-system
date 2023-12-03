from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

## edit below variables as per your requirements -
REPO_NAME = "Books-Recommender-System"
AUTHOR_USER_NAME = "BigData"
SRC_REPO = "src"
LIST_OF_REQUIREMENTS = ['streamlit', 'numpy']


setup(
    name=SRC_REPO,
    version="0.0.1",
    author=AUTHOR_USER_NAME,
    description="A package for Book Recommender System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/Felix-only/book-recommender-system",
    author_email="jl14865@nyu.edu",
    packages=[SRC_REPO],
    python_requires=">=3.7",
    install_requires=LIST_OF_REQUIREMENTS
)