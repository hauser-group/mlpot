import os
from setuptools.command.build_py import build_py
from setuptools import setup, find_packages
from subprocess import call

base_path = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(base_path, 'mlpot/descriptors')


class CustomBuild(build_py):
    """
    CustomBuild class following the suggestion in:
    https://stackoverflow.com/questions/1754966/how-can-i-run-a-makefile-in-setup-py
    """
    def run(self):

        def compile_library():
            call('make', cwd=lib_path)
        self.execute(compile_library, [],  'Compiling shared library')
        build_py.run(self)


setup(
    name='mlpot',
    version='0.2',
    description='Library for machine learning potentials',
    packages=find_packages(),
    package_data={'mlpot': ['descriptors/libDescriptorSet.so']},
    include_package_data=True,
    cmdclass={'build_py': CustomBuild}
)
