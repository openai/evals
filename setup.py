from setuptools import setup, find_packages

setup(
    name='evals',
    version='0.1.1',
    packages=find_packages(),
    install_requires=[
        'mypy',
        'openai',
        'tiktoken',
        'blobfile',
        'backoff',
        'numpy',
        'snowflake-connector-python[pandas]',
        'pandas',
        'fire',
        'pydantic',
        'tqdm',
        'nltk',
        'filelock',
        'mock',
        'langdetect',
        'termcolor',
        'lz4',
        'pyzstd',
        'pyyaml',
        'sacrebleu',
        'matplotlib',
    ],
    entry_points={
        'console_scripts': [
            'oaieval = evals.cli.oaieval:main',
            'oaievalset = evals.cli.oaievalset:main',
        ],
    },
)
