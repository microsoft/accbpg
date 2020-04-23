from setuptools import setup, find_packages

setup(
    name='accbpg',
    version='0.2',
    packages=find_packages(exclude=['tests*']),
    license='MIT',
    description='Accelerated Bregman proximal gradient (ABPG) methods',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=['numpy', 'scipy'],
    url='https://github.com/Microsoft/accbpg',
    author='Lin Xiao',
    author_email='lin.xiao@gmail.com'
)
