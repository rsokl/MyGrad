from setuptools import setup, find_packages


def do_setup():
    setup(name='MyGrad',
          version="0.0",
          author='YOUR NAME HERE',
          description='A pure-python autograd/tensor library',
          license='MIT',
          platforms=['Windows', 'Linux', 'Mac OS-X', 'Unix'],
          packages=find_packages(),
          install_requires=['numpy>=1.11'])

if __name__ == "__main__":
    do_setup()
