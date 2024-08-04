from setuptools import setup, find_packages

setup(
    name='IQGO_module',
    version='0.1.0',
    author='Your Name',
    author_email='d17125255@mytudublin.ie',
    description='IQGO: Iterative Quantum Gate Optimiser',
    long_description=open('README.md').read(),
    long_description_content_type='Classification Algorithm',
    url='https://github.com/copyright2010/IQGO-Iterative-Quantum-Gate-Optimiser',  # Replace with your repository URL
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)