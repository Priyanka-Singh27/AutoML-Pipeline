"""
narrator.py
Centralized UX narration module.
Formats and standardizes system outputs, suppressing unstructured logs.
"""
from core.headers import Section 

def narrate(*args, **kwargs):
    """
    Standardized console output wrapper.
    Replaces raw print() calls so all user-facing ML outputs 
    can be centrally stylized, routed, or bypassed later.
    """
    print(*args, **kwargs)
