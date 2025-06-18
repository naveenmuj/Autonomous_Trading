import os
import sys
import pytest

# Add tests directory to path
tests_dir = os.path.join(os.path.dirname(__file__), 'tests')
sys.path.insert(0, tests_dir)

# Import test configuration from tests/conftest.py
from conftest import *
