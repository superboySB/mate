"""MATE: The Multi-Agent Tracking Environment."""

import pathlib
import sys
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).absolute().parent

sys.path.insert(0, str(HERE / 'mate'))
import version  # pylint: disable=import-error,wrong-import-position

setup(
    name='mate',
    version=version.__version__,
    packages=find_packages(),  # 自动发现所有包
    include_package_data=True,  # 包含包内的所有文件
)
