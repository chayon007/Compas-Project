"""Transliteration-based adversarial attacks for Bangla text."""

import random
import numpy as np
from typing import Dict, List, Tuple, Optional
import re


def generate_bangla_to_roman_map() -> Dict[str, str]:
    """
    Generate Bangla to Roman phonetic transliteration map.
    
    This is a simplified phonetic mapping for common Bangla characters.
    """
    return {
        'অ': 'o',
        'আ': 'a',
        'ই': 'i',
        'ঈ': 'i',
        'উ': 'u',
        'ঊ': 'u',
        'ঋ': 'ri',
        'এ': 'e',
        'ঐ': 'ai',
        'ও': 'o',
        'ঔ': 'ou',
        'ক': 'k',
        'খ': 'kh',
        'গ': 'g',
        'ঘ': 'gh',
        'ঙ': 'ng',
        'চ': 'ch',
        'ছ': 'chh',
        'জ': 'j',
        'ঝ': 'jh',
        'ঞ': 'ny',
        'ট': 't',
        'ঠ': 'th',
        'ড': 'd',
        'ঢ': 'dh',
        'ণ': 'n',
        'ত': 't',
        'থ': 'th',
        'দ': 'd',
        'ধ': 'dh',
        'ন': 'n',
        'প': 'p',
        'ফ': 'ph',
        'ব': 'b',
        'ভ': 'bh',
        'ম': 'm',
        'য': 'y',
        'র': 'r',
        'ল': 'l',
        'শ': 'sh',
        'ষ': 'sh',
        'স': 's',
        'হ': 'h',
        'ড়': 'r',
        'ঢ়': 'rh',
        'য়': 'y',
        '়': '',
        'ৎ': 't',
        'ঃ': 'h',
        'ঁ': 'n',
    }


class TransliterationAttack:
    """Generate adversarial transliteration attacks."""
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize attack generator.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        self.bangla_to_roman = generate_bangla_to_roman_map()
    
    def transliterate_bangla_to_roman(self, text: str) -> str:
        """
        Transliterate Bangla text to Roman script.
        
        Args:
            text: Bangla text
        
        Returns:
            Romanized text
        """
        result = []
        for char in text:
            if char in self.bangla_to_roman:
                result.append(self.bangla_to_roman[char])
            else:
                # Keep non-Bangla characters as-is
                result.append(char)
        return ''.join(result)
    
    def level1_attack(self, text: str, perturbation_ratio: float = 0.2) -> str:
        """
        L1 Attack: Replace random words with Romanized versions.
        
        Args:
            text: Input text
            perturbation_ratio: Fraction of words to replace (default 0.2 = 20%)
        
        Returns:
            Perturbed text
        """
        words = text.split()
        num_to_perturb = max(1, int(len(words) * perturbation_ratio))
        
        # Randomly select indices to perturb
        indices_to_perturb = random.sample(range(len(words)), num_to_perturb)
        
        for idx in indices_to_perturb:
            words[idx] = self.transliterate_bangla_to_roman(words[idx])
        
        return ' '.join(words)
    
    def level2_attack(self, text: str) -> str:
        """
        L2 Attack: Full transliteration to Roman script.
        
        Args:
            text: Input text
        
        Returns:
            Fully transliterated text
        """
        return self.transliterate_bangla_to_roman(text)
    
    def level3_attack(self, text: str, code_mix_ratio: float = 0.5) -> str:
        """
        L3 Attack: Code-mixed attack (alternate Bangla and Roman tokens).
        
        Args:
            text: Input text
            code_mix_ratio: Fraction to transliterate (default 0.5 = 50%)
        
        Returns:
            Code-mixed text
        """
        words = text.split()
        num_to_transliterate = int(len(words) * code_mix_ratio)
        
        # Randomly select indices
        indices = random.sample(range(len(words)), num_to_transliterate)
        
        for idx in indices:
            words[idx] = self.transliterate_bangla_to_roman(words[idx])
        
        return ' '.join(words)
    
    def generate_attacked_dataset(
        self,
        texts: List[str],
        labels: List[int],
        attack_levels: List[str] = ['l1', 'l2', 'l3'],
        perturbation_ratio: float = 0.2
    ) -> Dict[str, Tuple[List[str], List[int]]]:
        """
        Generate adversarial versions of dataset for all attack levels.
        
        Args:
            texts: Original texts
            labels: Original labels
            attack_levels: Which attack levels to apply ['l1', 'l2', 'l3']
            perturbation_ratio: Perturbation strength for L1/L3
        
        Returns:
            Dictionary mapping attack type to (texts, labels) tuples
        """
        result = {
            'clean': (texts, labels)
        }
        
        if 'l1' in attack_levels:
            l1_texts = [self.level1_attack(t, perturbation_ratio) for t in texts]
            result['l1'] = (l1_texts, labels)
        
        if 'l2' in attack_levels:
            l2_texts = [self.level2_attack(t) for t in texts]
            result['l2'] = (l2_texts, labels)
        
        if 'l3' in attack_levels:
            l3_texts = [self.level3_attack(t) for t in texts]
            result['l3'] = (l3_texts, labels)
        
        return result
    
    def detect_script_type(self, text: str) -> str:
        """
        Detect script type in text: 'bangla', 'roman', or 'mixed'.
        """
        bangla_chars = sum(1 for c in text if '\u0980' <= c <= '\u09FF')
        roman_chars = sum(1 for c in text if ord(c) < 256 and c.isalpha())
        total = len(text)
        
        if total == 0:
            return 'empty'
        
        bangla_ratio = bangla_chars / total
        roman_ratio = roman_chars / total
        
        if bangla_ratio > 0.7:
            return 'bangla'
        elif roman_ratio > 0.7:
            return 'roman'
        else:
            return 'mixed'
