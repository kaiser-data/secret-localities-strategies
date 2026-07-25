"""Put `organism/` on sys.path so tests import `config` the same way the scripts do."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
