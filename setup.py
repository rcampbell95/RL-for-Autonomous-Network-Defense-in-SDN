import datetime
import setuptools

from pathlib import Path

VERSION_FILE = Path(__file__).resolve().parent.joinpath("version.txt")
NOW = datetime.datetime.utcnow()
VERSION = f'{NOW.year}.{NOW.month:02d}.{NOW.day:02d}'


with VERSION_FILE.open(mode="w") as f:
    f.write(VERSION)

setuptools.setup()
