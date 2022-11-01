
import sys
sys.path.insert(1, 'src')
from templates.utils import append_proper_article

simple_template = [
    lambda c: f"a photo of a {c}."
]