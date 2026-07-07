"""
Tests checking for vLLM upstream compatibility requirements.

As we remove support for old vLLM versions, we want to keep track of the
compatibility code that can be cleaned up.
"""

import pytest

pytestmark = pytest.mark.compat
