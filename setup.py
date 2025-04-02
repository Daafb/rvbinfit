from setuptools import setup, find_packages

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='rvbinfit',
      version='0.2',
      description='MCMC fitting of Binary RVs',
      long_description=readme(),
      url='https://github.com/gummiks/rvbinfit/',
      author='Gudmundur Stefansson',
      author_email='gummiks@gmail.com',
      install_requires=['emcee','batman-package','radvel','corner','pandas','pytransit'],
      packages=['rvbinfit'],
      license='GPLv3',
      classifiers=['Topic :: Scientific/Engineering :: Astronomy'],
      keywords='Astronomy',
      include_package_data=True
      )
