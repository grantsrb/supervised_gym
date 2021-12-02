from setuptools import setup, find_packages
from setuptools.command.install import install

setup(name='supervised_gym',
      packages=find_packages(),
      version="0.1.0",
      description='A project to train models to copy oracle gym players',
      author='Satchel Grant',
      author_email='grantsrb@stanford.edu',
      url='https://github.com/grantsrb/supervised_gym.git',
      install_requires= ["numpy",
                         "torch",
                         "tqdm"],
      py_modules=['supervised_gym'],
      long_description='''
            This project seeks to train models on OpenAI's gym games
            (or similarly styled games) by copying an oracle player.
            The oracle player must be created for the gym game by
            subclassing the Oracle class.
          ''',
      classifiers=[
          'Intended Audience :: Science/Research',
          'Operating System :: MacOS :: MacOS X :: Ubuntu',
          'Topic :: Scientific/Engineering :: Information Analysis'],
      )
