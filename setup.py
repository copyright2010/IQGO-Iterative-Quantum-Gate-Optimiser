from setuptools import setup, find_packages

setup(
    name='IQGO_module',
    version='0.1.0',
    author='Tautvydas Lisas',
    author_email='d17125255@mytudublin.ie',
    description='IQGO: Iterative Quantum Gate Optimiser',
    long_description=open('README.md').read(),
    long_description_content_type='Classification Algorithm',
    url='https://github.com/copyright2010/IQGO-Iterative-Quantum-Gate-Optimiser', 
    package_dir={'': 'src'},  # Add this line to specify the src directory
    packages=find_packages(where='src'),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)