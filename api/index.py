import os
import sys
from pathlib import Path

# Add the parent directory to sys.path so we can import app
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import app

# For Vercel, the WSGI app must be exposed as 'app'
# This is used by the Vercel Python Runtime
