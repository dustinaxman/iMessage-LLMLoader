from setuptools import setup, find_packages

setup(
    name="imessage-llmloader",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'vobject',
        'pandas',
        'matplotlib',
        'google-generativeai',
        'rich',
        'argparse',
	'tiktoken'
    ],
    author="Dustin Axman",
    author_email="dustinaxman@gmail.com",
    description="A package to load and process iMessages for input into an LLM",
)

