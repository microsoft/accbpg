from setuptools import setup, find_packages

setup(
    name='accbpg',
    version='0.1',
    packages=find_packages(exclude=['tests*']),
    license='MIT',
    description='Accelerated Bregman proximal gradient (ABPG) methods',
    long_description=open('README.md').read(),
    install_requires=['numpy'],
    url='https://github.com/Microsoft/accbpg',
    author='Lin Xiao',
    author_email='lin.xiao@gmail.com'
)
