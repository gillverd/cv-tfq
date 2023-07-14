from setuptools import setup, find_packages

setup(
    name='cv_tfq',
    version='0.0.1',
    description='Continuous Variable QC in TFQ',
    install_requires=['tensorflow_quantum', 'tensorflow'],
    packages=find_packages(include=['cv_tfq', 'cv_tfq.*'])
)