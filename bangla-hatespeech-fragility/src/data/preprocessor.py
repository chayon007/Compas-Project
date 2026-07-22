"""Text preprocessing and normalization utilities."""

import re
import unicodedata
from typing import Optional


class TextPreprocessor:
    """Text preprocessing for Bangla hate speech data."""
    
    def __init__(self, remove_urls: bool = False, remove_emails: bool = False,
                 normalize_unicode: bool = True, remove_extra_spaces: bool = True):
        """
        Initialize preprocessor.
        
        Args:
            remove_urls: Remove URLs from text
            remove_emails: Remove email addresses
            normalize_unicode: Normalize Unicode characters
            remove_extra_spaces: Remove extra whitespace
        """
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.normalize_unicode = normalize_unicode
        self.remove_extra_spaces = remove_extra_spaces
    
    def preprocess(self, text: str) -> str:
        """
        Apply preprocessing pipeline.
        
        Note: Minimal preprocessing to preserve hate speech signals (spellings, slurs, etc.)
        """
        if not isinstance(text, str):
            return ""
        
        text = str(text).strip()
        
        if self.normalize_unicode:
            text = self._normalize_unicode(text)
        
        if self.remove_urls:
            text = self._remove_urls(text)
        
        if self.remove_emails:
            text = self._remove_emails(text)
        
        if self.remove_extra_spaces:
            text = self._remove_extra_spaces(text)
        
        return text.strip()
    
    @staticmethod
    def _normalize_unicode(text: str) -> str:
        """Normalize Unicode to NFC form."""
        return unicodedata.normalize('NFC', text)
    
    @staticmethod
    def _remove_urls(text: str) -> str:
        """Remove URLs from text."""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.sub(url_pattern, '', text)
    
    @staticmethod
    def _remove_emails(text: str) -> str:
        """Remove email addresses from text."""
        email_pattern = r'\S+@\S+'
        return re.sub(email_pattern, '', text)
    
    @staticmethod
    def _remove_extra_spaces(text: str) -> str:
        """Remove extra whitespace."""
        return re.sub(r'\s+', ' ', text).strip()
    
    @staticmethod
    def get_script_type(text: str) -> str:
        """
        Detect script type: 'bangla', 'romanized', or 'mixed'.
        
        Bangla Unicode range: U+0980 to U+09FF
        """
        if not text:
            return 'unknown'
        
        bangla_count = sum(1 for c in text if '\u0980' <= c <= '\u09FF')
        roman_count = sum(1 for c in text if ord(c) < 256 and c.isalpha())
        total = len(text)
        
        if total == 0:
            return 'unknown'
        
        bangla_ratio = bangla_count / total
        roman_ratio = roman_count / total
        
        if bangla_ratio > 0.7:
            return 'bangla'
        elif roman_ratio > 0.7:
            return 'romanized'
        else:
            return 'mixed'
