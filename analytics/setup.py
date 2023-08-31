from setuptools import setup


setup(
    name='analytics',
    version='0.1',
    py_modules=['analytics'],
    install_requires=[
        'numpy>',
    ],
    tests_requires=[
        'pytest',
        'pathlib'
    ]
)
