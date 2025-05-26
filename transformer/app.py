# Add this at the VERY TOP of your transformer/app.py file, before any other imports:

import os
import sys

# CRITICAL: Fix PyTorch + Streamlit compatibility BEFORE any torch imports
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'
os.environ['STREAMLIT_SERVER_RUN_ON_SAVE'] = 'false'

# Suppress warnings early
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Fix PyTorch issues before importing
try:
    import torch
    torch.set_num_threads(1)  # Prevent threading issues
except ImportError:
    pass

import ssl
import random
import warnings
import re
import nltk
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from sentence_transformers import SentenceTransformer, util

warnings.filterwarnings("ignore", category=FutureWarning)

# Global NLP models for backward compatibility
NLP_GLOBAL = None
try:
    NLP_GLOBAL = spacy.load("en_core_web_sm")
except OSError:
    try:
        NLP_GLOBAL = spacy.load("fr_core_news_sm")
    except OSError:
        print("Warning: No spaCy models found. Please install at least one language model.")

        
def download_nltk_resources():
    """
    Download required NLTK resources if not already installed.
    """
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    resources = ['punkt', 'averaged_perceptron_tagger', 'punkt_tab', 'wordnet', 'averaged_perceptron_tagger_eng']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Error downloading {resource}: {str(e)}")


class MaritimeAcademicTextHumanizer:
    """
    Transforms text into a more formal academic style for maritime/naval science contexts.
    Supports both French and English with specialized maritime vocabulary and transitions.
    """

    def __init__(
        self,
        model_name='paraphrase-multilingual-MiniLM-L12-v2',
        p_passive=0.15,  # Reduced from 0.2
        p_synonym_replacement=0.25,  # Reduced from 0.3
        p_academic_transition=0.15,  # Reduced from 0.25
        p_maritime_terminology=0.3,  # Reduced from 0.4
        seed=None
    ):
        if seed is not None:
            random.seed(seed)

        # Load language models
        self.nlp_models = {}
        try:
            self.nlp_models['en'] = spacy.load("en_core_web_sm")
        except OSError:
            print("English spaCy model not found. Install with: python -m spacy download en_core_web_sm")
        
        try:
            self.nlp_models['fr'] = spacy.load("fr_core_news_sm")
        except OSError:
            print("French spaCy model not found. Install with: python -m spacy download fr_core_news_sm")

        if not self.nlp_models:
            raise Exception("No spaCy models available. Please install at least one language model.")

        self.model = SentenceTransformer(model_name)

        # Transformation probabilities
        self.p_passive = p_passive
        self.p_synonym_replacement = p_synonym_replacement
        self.p_academic_transition = p_academic_transition
        self.p_maritime_terminology = p_maritime_terminology

        # Context-aware academic transitions by language
        self.academic_transitions = {
            'en': {
                'neutral': ["Moreover,", "Furthermore,", "Additionally,", "Hence,", "Therefore,", "Consequently,", "Nevertheless,", "However,"],
                'maritime': ["In maritime operations,", "From a maritime perspective,", "In commercial navigation,", "According to maritime regulations,", "Considering vessel operations,"],
                'technical': ["With regard to shipboard procedures,", "In line with SOLAS conventions,", "As per international maritime standards,", "Regarding operational safety measures,"]
            },
            'fr': {
                'neutral': ["Par ailleurs,", "En outre,", "De plus,", "Cependant,", "Néanmoins,", "Ainsi,", "Par conséquent,", "En effet,"],
                'maritime': ["Dans le contexte maritime,", "Du point de vue maritime,", "En navigation commerciale,", "Selon la réglementation maritime,", "Concernant les opérations navales,"],
                'technical': ["En conformité avec la réglementation SOLAS,", "D'un point de vue technique,","Relativement aux procédures,"]
            }
        }

        # Enhanced maritime-specific academic synonyms with context checking
        self.maritime_synonyms = {
            'en': {
                # General academic terms
                'important': ['significant', 'crucial', 'vital', 'critical', 'essential'],
                'big': ['substantial', 'considerable', 'extensive', 'major'],
                'small': ['minimal', 'limited', 'modest', 'minor'],
                'show': ['demonstrate', 'illustrate', 'reveal', 'indicate', 'exhibit'],
                'use': ['utilize', 'employ', 'implement', 'apply'],
                'make': ['construct', 'fabricate', 'manufacture', 'produce'],
                'study': ['investigate', 'examine', 'analyze', 'assess', 'evaluate'],
                
                # Maritime-specific terms
                'ship': ['vessel', 'craft'],
                'boat': ['vessel', 'craft', 'watercraft'],
                'sea': ['marine environment', 'navigable waters', 'maritime domain'],
                'ocean': ['marine environment', 'oceanic waters', 'maritime domain'],
                'wave': ['sea state', 'marine oscillation'],
                'wind': ['meteorological factor', 'atmospheric condition'],
                'move': ['navigate', 'proceed', 'maneuver'],
                'fast': ['high-speed', 'rapid'],
                'slow': ['reduced-speed', 'low-velocity'],
                'safe': ['secure', 'compliant with safety standards'],
                'dangerous': ['hazardous', 'non-compliant', 'unsafe']
            },
            'fr': {
                # General academic terms
                'important': ['significatif', 'crucial', 'essentiel', 'majeur', 'primordial'],
                'grand': ['considérable', 'substantiel', 'important', 'majeur'],
                'petit': ['modeste', 'limité', 'restreint', 'mineur'],
                'montrer': ['démontrer', 'illustrer', 'révéler', 'indiquer'],
                'utiliser': ['employer', 'mettre en œuvre', 'appliquer'],
                'faire': ['effectuer', 'réaliser', 'accomplir', 'exécuter'],
                'étudier': ['examiner', 'analyser', 'évaluer', 'investiguer'],

                # Maritime-specific terms
                'bateau': ['navire', 'bâtiment'],
                
                'mer': ['domaine marin', 'environnement maritime', 'espace maritime'],
                'océan': ['domaine océanique', 'espace maritime'],
                'vague': ['état de la mer', 'houle', 'oscillation marine'],
                'vent': ['facteur météorologique', 'condition atmosphérique'],
                'bouger': ['naviguer', 'manœuvrer', 'évoluer'],
                'rapide': ['à haute vitesse', 'véloce'],
                'lent': ['à vitesse réduite', 'lente'],
                'sûr': ['sécurisé', 'conforme aux normes de sécurité'],
                'dangereux': ['risqué', 'non conforme', 'périlleux']
            }
        }

        # Contractions by language
        self.contractions = {
            'en': {
                "n't": " not", "'re": " are", "'s": " is", "'ll": " will",
                "'ve": " have", "'d": " would", "'m": " am"
            },
            'fr': {
                "j'suis": "je suis", "j'peux": "je peux", "j'veux": "je veux",
                "t'as": "tu as", "t'es": "tu es", "y'a": "il y a",
                "d'abord": "d'abord", "aujourd'hui": "aujourd'hui"  # Keep these as-is
            }
        }

        # Keywords to detect maritime context
        self.maritime_keywords = {
            'en': ['ship', 'vessel', 'boat', 'sea', 'ocean', 'navigation', 'maritime', 'naval', 'port', 'harbor', 'crew', 'captain', 'engine', 'deck', 'cargo', 'fuel', 'anchor', 'bridge', 'helm'],
            'fr': ['navire', 'bateau', 'mer', 'océan', 'navigation', 'maritime', 'naval', 'port', 'équipage', 'capitaine', 'moteur', 'pont', 'cargaison', 'carburant', 'ancre', 'passerelle', 'barre']
        }

    def detect_language(self, text):
        """
        Improved language detection with better accuracy
        """
        # Comprehensive language indicators
        french_indicators = [
            'le', 'la', 'les', 'de', 'des', 'du', 'et', 'est', 'dans', 'pour', 
            'avec', 'sur', 'par', 'navire', 'mer', 'que', 'qui', 'une', 'un',
            'ce', 'cette', 'ces', 'sont', 'être', 'avoir', 'faire', 'aller'
        ]
        english_indicators = [
            'the', 'and', 'is', 'in', 'for', 'with', 'on', 'by', 'ship', 'sea', 
            'vessel', 'water', 'of', 'to', 'a', 'an', 'this', 'that', 'these',
            'are', 'be', 'have', 'do', 'go', 'will', 'would', 'could'
        ]
        
        words = re.findall(r'\b\w+\b', text.lower())
        french_count = sum(1 for word in words if word in french_indicators)
        english_count = sum(1 for word in words if word in english_indicators)
        
        # Add weight for language-specific patterns
        if re.search(r'\b(qu|c\'est|n\'est|d\'un|l\')\b', text.lower()):
            french_count += 2
        if re.search(r'\b(it\'s|don\'t|can\'t|won\'t)\b', text.lower()):
            english_count += 2
            
        return 'fr' if french_count > english_count else 'en'

    def detect_maritime_context(self, text, language):
        """
        Detect if the text has maritime context
        """
        keywords = self.maritime_keywords.get(language, [])
        text_lower = text.lower()
        maritime_score = sum(1 for keyword in keywords if keyword in text_lower)
        return maritime_score > 0

    def get_context_type(self, sentence, language):
        """
        Determine the context type of a sentence for appropriate transition selection
        """
        sentence_lower = sentence.lower()
        maritime_keywords = self.maritime_keywords.get(language, [])
        
        # Check for maritime context
        if any(keyword in sentence_lower for keyword in maritime_keywords):
            # Check for technical terms
            technical_terms = {
                'en': ['system', 'procedure', 'regulation', 'standard', 'protocol', 'specification'],
                'fr': ['système', 'procédure', 'réglementation', 'norme', 'protocole', 'spécification']
            }
            if any(term in sentence_lower for term in technical_terms.get(language, [])):
                return 'technical'
            return 'maritime'
        
        return 'neutral'

    def humanize_text(self, text, language=None, use_passive=False, use_synonyms=True, use_maritime_terms=True):
        """
        Humanize text with maritime academic style and improved context awareness
        """
        if language is None:
            language = self.detect_language(text)
        
        if language not in self.nlp_models:
            print(f"Warning: {language} model not available. Using available model.")
            language = list(self.nlp_models.keys())[0]

        nlp = self.nlp_models[language]
        
        # Check overall maritime context
        has_maritime_context = self.detect_maritime_context(text, language)
        
        try:
            doc = nlp(text)
        except Exception as e:
            print(f"Error processing text with spaCy: {e}")
            return text
            
        transformed_sentences = []

        for i, sent in enumerate(doc.sents):
            sentence_str = sent.text.strip()
            
            if not sentence_str:
                continue

            # 1. Expand contractions
            sentence_str = self.expand_contractions(sentence_str, language)

            # 2. Add academic transitions (only for non-first sentences and with context awareness)
            if i > 0 and random.random() < self.p_academic_transition:
                context_type = self.get_context_type(sentence_str, language)
                sentence_str = self.add_academic_transitions(sentence_str, language, context_type)

            # 3. Convert to passive voice (reduced probability)
            if use_passive and random.random() < self.p_passive:
                sentence_str = self.convert_to_passive(sentence_str, language)

            # 4. Replace with synonyms (context-aware)
            if use_synonyms and random.random() < self.p_synonym_replacement:
                sentence_str = self.replace_with_synonyms(sentence_str, language, has_maritime_context)

            # 5. Enhance maritime terminology (only if maritime context detected)
            if use_maritime_terms and has_maritime_context and random.random() < self.p_maritime_terminology:
                sentence_str = self.enhance_maritime_terminology(sentence_str, language)

            transformed_sentences.append(sentence_str)

        return ' '.join(transformed_sentences)

    def expand_contractions(self, sentence, language):
        """
        Expand contractions based on language with improved French handling
        """
        contraction_map = self.contractions.get(language, {})
        
        if language == 'en':
            tokens = word_tokenize(sentence)
            expanded_tokens = []
            for token in tokens:
                lower_token = token.lower()
                replaced = False
                for contraction, expansion in contraction_map.items():
                    if lower_token.endswith(contraction):
                        new_token = lower_token.replace(contraction, expansion)
                        if token[0].isupper():
                            new_token = new_token.capitalize()
                        expanded_tokens.append(new_token)
                        replaced = True
                        break
                if not replaced:
                    expanded_tokens.append(token)
            return ' '.join(expanded_tokens)
        
        elif language == 'fr':
            expanded = sentence
            # Be more careful with French contractions
            for contraction, expansion in contraction_map.items():
                if contraction in expanded.lower():
                    expanded = re.sub(r'\b' + re.escape(contraction) + r'\b', expansion, expanded, flags=re.IGNORECASE)
            return expanded
        
        return sentence

    def add_academic_transitions(self, sentence, language, context_type='neutral'):
        """
        Add context-appropriate academic transitions
        """
        transitions_dict = self.academic_transitions.get(language, {})
        transitions = transitions_dict.get(context_type, transitions_dict.get('neutral', []))
        
        if transitions and sentence:
            transition = random.choice(transitions)
            # Ensure proper capitalization
            if sentence[0].isupper():
                if language == 'fr':
                    return f"{transition} {sentence[0].lower() + sentence[1:]}"
                else:
                    return f"{transition} {sentence[0].lower() + sentence[1:]}"
            return f"{transition} {sentence}"
        return sentence

    def convert_to_passive(self, sentence, language):
        """
        Convert to passive voice with improved accuracy and context checking
        """
        if language not in self.nlp_models:
            return sentence
            
        nlp = self.nlp_models[language]
        
        try:
            doc = nlp(sentence)
        except:
            return sentence
        
        # Only attempt conversion for sentences with clear subject-verb-object structure
        subjects = [token for token in doc if token.dep_ == "nsubj" and token.pos_ in ["NOUN", "PRON", "PROPN"]]
        objects = [token for token in doc if token.dep_ in ["obj", "dobj"] and token.pos_ in ["NOUN", "PROPN"]]
        
        if subjects and objects and len(subjects) == 1 and len(objects) == 1:
            subj = subjects[0]
            obj = objects[0]
            verb = subj.head
            
            # Only convert simple present/past tense verbs
            if language == 'en' and verb.tag_ in ['VBD', 'VBZ', 'VBP']:
                past_participle = self._get_past_participle_en(verb.lemma_)
                if past_participle:
                    be_verb = "was" if verb.tag_ == 'VBD' else "is"
                    passive_form = f"{obj.text} {be_verb} {past_participle} by {subj.text}"
                    return sentence.replace(f"{subj.text} {verb.text} {obj.text}", passive_form, 1)
            
            elif language == 'fr' and verb.pos_ == "VERB":
                participle = self._get_past_participle_fr(verb.lemma_)
                if participle:
                    auxiliary = "est" if obj.tag_ in ["NOUN", "PROPN"] else "sont"
                    passive_form = f"{obj.text} {auxiliary} {participle} par {subj.text}"
                    return sentence.replace(f"{subj.text} {verb.text} {obj.text}", passive_form, 1)
        
        return sentence

    def replace_with_synonyms(self, sentence, language, has_maritime_context=False):
        """
        Replace words with academic synonyms with context awareness
        """
        synonyms_dict = self.maritime_synonyms.get(language, {})
        
        try:
            tokens = word_tokenize(sentence)
        except:
            tokens = sentence.split()
            
        new_tokens = []
        
        for token in tokens:
            token_lower = token.lower().strip('.,!?;:')
            
            if token_lower in synonyms_dict:
                # Reduce replacement probability to avoid over-transformation
                if random.random() < 0.3:  # Reduced from 0.4
                    synonyms = synonyms_dict[token_lower]
                    # For maritime context, prefer maritime-specific synonyms
                    if has_maritime_context and len(synonyms) > 2:
                        synonym = random.choice(synonyms[:2])  # Use first 2 (usually more specific)
                    else:
                        synonym = random.choice(synonyms)
                    
                    # Preserve capitalization
                    if token[0].isupper():
                        synonym = synonym.capitalize()
                    
                    # Preserve punctuation
                    punctuation = ''.join(c for c in token if not c.isalnum())
                    new_tokens.append(synonym + punctuation)
                else:
                    new_tokens.append(token)
            else:
                new_tokens.append(token)
        
        return ' '.join(new_tokens)

    def enhance_maritime_terminology(self, sentence, language):
        """
        Enhanced maritime terminology with better context awareness
        """
        maritime_enhancements = {
            'en': {
                'stability': 'vessel stability characteristics',
                'movement': 'vessel dynamics',
                'speed': 'operational velocity', 
                'navigation': 'integrated navigation system',
                'safety': 'maritime safety protocols',
                'communication': 'ship-to-shore communications',
                'engine': 'propulsion system',
                'fuel': 'marine fuel system',
                'crew': 'shipboard personnel'
            },
            'fr': {
                'stabilité': 'caractéristiques de stabilité du navire',
                'mouvement': 'dynamique du navire',
                'vitesse': 'vitesse opérationnelle',
                'navigation': 'système de navigation intégré',
                'sécurité': 'protocoles de sécurité maritime',
                'communication': 'communications navire-terre',
                'moteur': 'système de propulsion',
                'carburant': 'système de carburant marin',
                'équipage': 'personnel embarqué'
            }
        }
        
        enhancements = maritime_enhancements.get(language, {})
        
        for term, enhancement in enhancements.items():
            # Only replace if the term appears as a standalone word and with lower probability
            if re.search(r'\b' + re.escape(term) + r'\b', sentence.lower()) and random.random() < 0.2:
                pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
                sentence = pattern.sub(enhancement, sentence, count=1)
                break  # Only one replacement per sentence
        
        return sentence

    def _get_past_participle_en(self, verb_lemma):
        """
        Get English past participle (enhanced)
        """
        participles = {
            'make': 'made', 'take': 'taken', 'give': 'given', 'see': 'seen',
            'do': 'done', 'go': 'gone', 'come': 'come', 'know': 'known',
            'get': 'gotten', 'find': 'found', 'think': 'thought',
            'say': 'said', 'tell': 'told', 'use': 'used', 'show': 'shown',
            'write': 'written', 'read': 'read', 'hear': 'heard', 'feel': 'felt'
        }
        return participles.get(verb_lemma, verb_lemma + 'ed')

    def _get_past_participle_fr(self, verb_lemma):
        """
        Get French past participle (enhanced)
        """
        participles = {
            'faire': 'fait', 'dire': 'dit', 'voir': 'vu', 'prendre': 'pris',
            'donner': 'donné', 'mettre': 'mis', 'écrire': 'écrit',
            'lire': 'lu', 'comprendre': 'compris', 'naviguer': 'navigué',
            'manœuvrer': 'manœuvré', 'analyser': 'analysé', 'utiliser': 'utilisé',
            'effectuer': 'effectué', 'réaliser': 'réalisé'
        }
        return participles.get(verb_lemma)


# Enhanced convenience functions
def process_maritime_text(text, language=None, conservative=True):
    """
    Convenience function for processing maritime academic text with conservative settings
    """
    if conservative:
        humanizer = MaritimeAcademicTextHumanizer(
            p_passive=0.1,
            p_synonym_replacement=0.2,
            p_academic_transition=0.1,
            p_maritime_terminology=0.25,
            seed=42
        )
    else:
        humanizer = MaritimeAcademicTextHumanizer(seed=42)
    
    return humanizer.humanize_text(
        text, 
        language=language, 
        use_synonyms=True, 
        use_maritime_terms=True
    )

def process_naval_report(text, language=None, formal_level='medium'):
    """
    Process text for naval/maritime reports with different formality levels
    """
    if formal_level == 'high':
        p_synonym = 0.35
        p_transition = 0.2
        p_maritime = 0.4
        p_passive = 0.15
    elif formal_level == 'medium':
        p_synonym = 0.25
        p_transition = 0.15
        p_maritime = 0.3
        p_passive = 0.1
    else:  # low
        p_synonym = 0.15
        p_transition = 0.1
        p_maritime = 0.2
        p_passive = 0.05
    
    humanizer = MaritimeAcademicTextHumanizer(
        p_synonym_replacement=p_synonym,
        p_academic_transition=p_transition,
        p_maritime_terminology=p_maritime,
        p_passive=p_passive,
        seed=42
    )
    
    return humanizer.humanize_text(
        text,
        language=language,
        use_synonyms=True,
        use_maritime_terms=True,
        use_passive=True
    )