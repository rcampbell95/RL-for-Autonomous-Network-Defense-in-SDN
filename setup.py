import datetime
import setuptools

from pathlib import Path

REQ_FILE = Path(__file__).resolve().parent.joinpath('requirements.txt')
NOW = datetime.datetime.utcnow()
VERSION = f'{NOW.year}.{NOW.month:02d}.{NOW.day:02d}'


with REQ_FILE.open(mode='r') as fin:
    REQS = [line.strip() for line in fin]


setuptools.setup(
    install_requires=REQS,
    version=VERSION
)
