from setuptools import setup, Extension, find_packages
import setuptools

setup(
  name = "pynumbuf",
  version = "0.1.dev0",
  packages=find_packages(),
  package_data = {
    "pynumbuf": ["libpynumbuf.so"]
  },
  zip_safe=False
)
