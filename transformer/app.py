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
    Transforms text into a more formal academic style with improved contextual awareness.
    Supports both French and English with context-appropriate vocabulary and transitions.
    """

    def __init__(
        self,
        model_name='paraphrase-multilingual-MiniLM-L12-v2',
        p_passive=0.15,
        p_synonym_replacement=0.25,
        p_academic_transition=0.15,
        p_maritime_terminology=0.3,
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

        # Improved academic transitions - removed maritime-specific ones to avoid out-of-context usage
        self.academic_transitions = {
            'en': {
                'beginning': ["Furthermore,", "Additionally,", "Moreover,"],
                'contrast': ["However,", "Nevertheless,", "Nonetheless,", "On the other hand,"],
                'result': ["Therefore,", "Consequently,", "As a result,", "Hence,"],
                'emphasis': ["Indeed,", "In fact,", "Notably,", "Significantly,"],
                'neutral': ["Furthermore,", "Additionally,", "Moreover,", "However,", "Therefore,", "Nevertheless,"]
            },
            'fr': {
                'beginning': ["En outre,", "De plus,", "Par ailleurs,"],
                'contrast': ["Cependant,", "Néanmoins,", "Toutefois,", "D'autre part,"],
                'result': ["Par conséquent,", "Ainsi,", "De ce fait,", "En conséquence,"],
                'emphasis': ["En effet,", "De fait,", "Notamment,", "Il convient de noter que"],
                'neutral': ["En outre,", "De plus,", "Cependant,", "Par conséquent,", "Néanmoins,", "Ainsi,"]
            }
        }

        # Context-aware academic synonyms (removed overly specific maritime terms)
        self.academic_synonyms = {
            'en': {
                # General academic terms
                'important': ['significant', 'crucial', 'vital', 'essential', 'critical'],
                'big': ['substantial', 'considerable', 'extensive', 'major', 'large-scale'],
                'small': ['minimal', 'limited', 'modest', 'minor', 'restricted'],
                'show': ['demonstrate', 'illustrate', 'reveal', 'indicate', 'exhibit'],
                'use': ['utilize', 'employ', 'implement', 'apply', 'adopt'],
                'make': ['create', 'produce', 'construct', 'generate', 'establish'],
                'study': ['investigate', 'examine', 'analyze', 'assess', 'evaluate'],
                'think': ['consider', 'believe', 'assume', 'hypothesize', 'postulate'],
                'help': ['assist', 'facilitate', 'support', 'contribute to', 'enhance'],
                'good': ['effective', 'beneficial', 'advantageous', 'positive', 'favorable'],
                'bad': ['detrimental', 'adverse', 'negative', 'unfavorable', 'problematic'],
                'many': ['numerous', 'multiple', 'various', 'several', 'abundant'],
                'very': ['extremely', 'significantly', 'considerably', 'substantially', 'remarkably'],
                'find': ['discover', 'identify', 'determine', 'establish', 'ascertain'],
                'get': ['obtain', 'acquire', 'secure', 'achieve', 'attain'],
                
                # Maritime context - only when maritime keywords are present
                'ship': ['vessel', 'craft', 'maritime vehicle'],
                'boat': ['vessel', 'watercraft', 'craft'],
                'move': ['navigate', 'proceed', 'advance', 'progress'],
                'fast': ['rapid', 'swift', 'high-speed'],
                'slow': ['gradual', 'steady', 'measured'],
                'safe': ['secure', 'protected', 'reliable'],
                'dangerous': ['hazardous', 'risky', 'perilous']
            },
            'fr': {
                # General academic terms
                'important': ['significatif', 'crucial', 'essentiel', 'majeur', 'fondamental'],
                'grand': ['considérable', 'substantiel', 'important', 'majeur', 'étendu'],
                'petit': ['modeste', 'limité', 'restreint', 'mineur', 'réduit'],
                'montrer': ['démontrer', 'illustrer', 'révéler', 'indiquer', 'présenter'],
                'utiliser': ['employer', 'mettre en œuvre', 'appliquer', 'adopter', 'recourir à'],
                'faire': ['effectuer', 'réaliser', 'accomplir', 'exécuter', 'créer'],
                'étudier': ['examiner', 'analyser', 'évaluer', 'investiguer', 'explorer'],
                'penser': ['considérer', 'estimer', 'supposer', 'présumer', 'envisager'],
                'aider': ['assister', 'faciliter', 'soutenir', 'contribuer à', 'favoriser'],
                'bon': ['efficace', 'bénéfique', 'avantageux', 'positif', 'favorable'],
                'mauvais': ['néfaste', 'défavorable', 'négatif', 'problématique', 'préjudiciable'],
                'beaucoup': ['nombreux', 'multiples', 'divers', 'plusieurs', 'abondants'],
                'très': ['extrêmement', 'considérablement', 'remarquablement', 'particulièrement'],
                'trouver': ['découvrir', 'identifier', 'déterminer', 'établir', 'constater'],
                'avoir': ['obtenir', 'acquérir', 'posséder', 'disposer de', 'bénéficier de'],
                
                # Maritime context - only when maritime keywords are present
                'bateau': ['navire', 'bâtiment'],
                
                'mer': ['domaine marin', 'environnement maritime', 'espace maritime'],
                'océan': ['domaine océanique', 'espace maritime'],
                'vague': ['état de la mer', 'houle', 'oscillation marine'],
                'vent': ['facteur météorologique', 'condition atmosphérique'],

                'navire': ['bâtiment', 'vaisseau', 'unité navale'],
                'bouger': ['naviguer', 'évoluer', 'progresser', 'avancer'],
                'rapide': ['véloce', 'swift', 'accéléré'],
                'lent': ['graduel', 'mesuré', 'progressif'],
                'sûr': ['sécurisé', 'fiable', 'protégé'],
                'dangereux': ['risqué', 'périlleux', 'hasardeux']
            }
        }

        # Contractions by language
        self.contractions = {
            'en': {
                "n't": " not", "'re": " are", "'s": " is", "'ll": " will",
                "'ve": " have", "'d": " would", "'m": " am", "'t": " not"
            },
            'fr': {
                "j'suis": "je suis", "j'peux": "je peux", "j'veux": "je veux",
                "t'as": "tu as", "t'es": "tu es", "y'a": "il y a",
                "c'est": "cela est", "qu'est-ce": "que est-ce"
            }
        }

        # Keywords to detect maritime context (more comprehensive)
        self.maritime_keywords = {
            'en': [
                'ship', 'vessel', 'boat', 'sea', 'ocean', 'navigation', 'maritime', 'naval', 
                'port', 'harbor', 'crew', 'captain', 'engine', 'deck', 'cargo', 'fuel', 
                'anchor', 'bridge', 'helm', 'sailing', 'marine', 'nautical', 'shipboard',
                'fleet', 'dock', 'pier', 'lighthouse', 'buoy', 'chart', 'compass', 'radar'
            ],
            'fr': [
                'navire', 'bateau', 'mer', 'océan', 'navigation', 'maritime', 'naval', 
                'port', 'équipage', 'capitaine', 'moteur', 'pont', 'cargaison', 'carburant', 
                'ancre', 'passerelle', 'barre', 'voile', 'marin', 'nautique', 'bord',
                'flotte', 'quai', 'jetée', 'phare', 'bouée', 'carte', 'boussole', 'radar'
            ]
        }

        # Sentence context patterns for better transition selection
        self.context_patterns = {
            'en': {
                'contrast': ['but', 'however', 'although', 'despite', 'while', 'whereas'],
                'result': ['because', 'since', 'therefore', 'so', 'thus', 'hence'],
                'addition': ['and', 'also', 'additionally', 'furthermore', 'moreover'],
                'emphasis': ['important', 'significant', 'crucial', 'key', 'main', 'primary']
            },
            'fr': {
                'contrast': ['mais', 'cependant', 'bien que', 'malgré', 'tandis que', 'alors que'],
                'result': ['parce que', 'puisque', 'donc', 'ainsi', 'par conséquent'],
                'addition': ['et', 'aussi', 'également', 'de plus', 'en outre'],
                'emphasis': ['important', 'significatif', 'crucial', 'clé', 'principal', 'primordial']
            }
        }

    def detect_language(self, text):
        """
        Improved language detection with better accuracy
        """
        french_indicators = [
            'le', 'la', 'les', 'de', 'des', 'du', 'et', 'est', 'dans', 'pour', 
            'avec', 'sur', 'par', 'que', 'qui', 'une', 'un', 'ce', 'cette', 
            'ces', 'sont', 'être', 'avoir', 'faire', 'aller', 'voir', 'savoir'
        ]
        english_indicators = [
            'the', 'and', 'is', 'in', 'for', 'with', 'on', 'by', 'of', 'to', 
            'a', 'an', 'this', 'that', 'these', 'are', 'be', 'have', 'do', 
            'go', 'will', 'would', 'could', 'can', 'should', 'may', 'might'
        ]
        
        words = re.findall(r'\b\w+\b', text.lower())
        french_count = sum(1 for word in words if word in french_indicators)
        english_count = sum(1 for word in words if word in english_indicators)
        
        # Add weight for language-specific patterns
        if re.search(r'\b(qu\'|c\'est|n\'est|d\'un|l\'|j\')\b', text.lower()):
            french_count += 3
        if re.search(r'\b(it\'s|don\'t|can\'t|won\'t|I\'m|you\'re)\b', text.lower()):
            english_count += 3
            
        return 'fr' if french_count > english_count else 'en'

    def detect_maritime_context(self, text, language):
        """
        Detect if the text has maritime context with minimum threshold
        """
        keywords = self.maritime_keywords.get(language, [])
        text_lower = text.lower()
        maritime_score = sum(1 for keyword in keywords if keyword in text_lower)
        # Require at least 2 maritime keywords for context detection
        return maritime_score >= 2

    def get_sentence_context(self, sentence, language, previous_sentence=""):
        """
        Determine the context type of a sentence for appropriate transition selection
        """
        sentence_lower = sentence.lower()
        prev_lower = previous_sentence.lower() if previous_sentence else ""
        
        patterns = self.context_patterns.get(language, {})
        
        # Check for contrast indicators
        if any(word in sentence_lower or word in prev_lower for word in patterns.get('contrast', [])):
            return 'contrast'
        
        # Check for result/conclusion indicators  
        if any(word in sentence_lower or word in prev_lower for word in patterns.get('result', [])):
            return 'result'
            
        # Check for emphasis indicators
        if any(word in sentence_lower for word in patterns.get('emphasis', [])):
            return 'emphasis'
            
        # Default to neutral beginning transition
        return 'beginning'

    def humanize_text(self, text, language=None, use_passive=False, use_synonyms=True, use_maritime_terms=True):
        """
        Humanize text with improved context awareness and reduced over-transformation
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
            
        sentences = list(doc.sents)
        transformed_sentences = []

        for i, sent in enumerate(sentences):
            sentence_str = sent.text.strip()
            
            if not sentence_str:
                continue

            # 1. Expand contractions
            sentence_str = self.expand_contractions(sentence_str, language)

            # 2. Add academic transitions (only for non-first sentences with proper context)
            if i > 0 and random.random() < self.p_academic_transition:
                previous_sentence = sentences[i-1].text if i > 0 else ""
                context_type = self.get_sentence_context(sentence_str, language, previous_sentence)
                sentence_str = self.add_contextual_transitions(sentence_str, language, context_type)

            # 3. Convert to passive voice (only for appropriate sentences)
            if use_passive and random.random() < self.p_passive:
                sentence_str = self.convert_to_passive(sentence_str, language)

            # 4. Replace with synonyms (context-aware and more conservative)
            if use_synonyms and random.random() < self.p_synonym_replacement:
                sentence_str = self.replace_with_synonyms(sentence_str, language, has_maritime_context)

            # 5. Enhance terminology only if maritime context is clearly present
            if use_maritime_terms and has_maritime_context and random.random() < (self.p_maritime_terminology * 0.7):
                sentence_str = self.enhance_terminology(sentence_str, language)

            transformed_sentences.append(sentence_str)

        return ' '.join(transformed_sentences)

    def expand_contractions(self, sentence, language):
        """
        Expand contractions based on language with improved accuracy
        """
        contraction_map = self.contractions.get(language, {})
        
        if language == 'en':
            # More precise contraction handling
            for contraction, expansion in contraction_map.items():
                pattern = r'\b\w+' + re.escape(contraction) + r'\b'
                matches = re.finditer(pattern, sentence, re.IGNORECASE)
                for match in reversed(list(matches)):
                    word = match.group()
                    base_word = word[:-len(contraction)]
                    new_word = base_word + expansion
                    # Preserve original capitalization
                    if word[0].isupper():
                        new_word = new_word[0].upper() + new_word[1:]
                    sentence = sentence[:match.start()] + new_word + sentence[match.end():]
        
        elif language == 'fr':
            # Handle French contractions more carefully
            for contraction, expansion in contraction_map.items():
                sentence = re.sub(r'\b' + re.escape(contraction) + r'\b', expansion, sentence, flags=re.IGNORECASE)
        
        return sentence

    def add_contextual_transitions(self, sentence, language, context_type):
        """
        Add context-appropriate academic transitions
        """
        transitions_dict = self.academic_transitions.get(language, {})
        transitions = transitions_dict.get(context_type, transitions_dict.get('neutral', []))
        
        if transitions and sentence:
            transition = random.choice(transitions)
            # Ensure proper sentence structure
            if sentence[0].isupper():
                return f"{transition} {sentence[0].lower() + sentence[1:]}"
            return f"{transition} {sentence}"
        return sentence

    def convert_to_passive(self, sentence, language):
        """
        Convert to passive voice with better accuracy and context checking
        """
        if language not in self.nlp_models:
            return sentence
            
        nlp = self.nlp_models[language]
        
        try:
            doc = nlp(sentence)
        except:
            return sentence
        
        # Only attempt conversion for sentences with clear structure
        subjects = [token for token in doc if token.dep_ == "nsubj" and token.pos_ in ["NOUN", "PRON", "PROPN"]]
        objects = [token for token in doc if token.dep_ in ["obj", "dobj"] and token.pos_ in ["NOUN", "PROPN"]]
        
        # Be more selective about passive voice conversion
        if subjects and objects and len(subjects) == 1 and len(objects) == 1:
            subj = subjects[0]
            obj = objects[0]
            verb = subj.head
            
            # Check if conversion makes sense contextually
            if self._should_convert_to_passive(sentence, subj.text, verb.text, obj.text):
                if language == 'en' and verb.tag_ in ['VBD', 'VBZ', 'VBP']:
                    past_participle = self._get_past_participle_en(verb.lemma_)
                    if past_participle:
                        be_verb = "was" if verb.tag_ == 'VBD' else "is"
                        return sentence.replace(f"{subj.text} {verb.text} {obj.text}", 
                                              f"{obj.text} {be_verb} {past_participle} by {subj.text}", 1)
                
                elif language == 'fr' and verb.pos_ == "VERB":
                    participle = self._get_past_participle_fr(verb.lemma_)
                    if participle:
                        auxiliary = "est" if obj.tag_ in ["NOUN", "PROPN"] else "sont"
                        return sentence.replace(f"{subj.text} {verb.text} {obj.text}", 
                                              f"{obj.text} {auxiliary} {participle} par {subj.text}", 1)
        
        return sentence

    def _should_convert_to_passive(self, sentence, subject, verb, obj):
        """
        Determine if a sentence should be converted to passive voice
        """
        # Avoid converting questions, imperatives, or very short sentences
        if len(sentence.split()) < 4:
            return False
        if sentence.strip().endswith('?') or sentence.strip().startswith(('What', 'When', 'Where', 'How', 'Why')):
            return False
        # Avoid converting sentences with pronouns as subjects (less formal in passive)
        if subject.lower() in ['i', 'you', 'we', 'they', 'je', 'tu', 'nous', 'vous', 'ils', 'elles']:
            return False
        return True

    def replace_with_synonyms(self, sentence, language, has_maritime_context=False):
        """
        Replace words with academic synonyms with improved context awareness
        """
        synonyms_dict = self.academic_synonyms.get(language, {})
        
        try:
            tokens = word_tokenize(sentence)
        except:
            tokens = sentence.split()
            
        new_tokens = []
        words_replaced = 0
        max_replacements = max(1, len(tokens) // 4)  # Limit replacements per sentence
        
        for token in tokens:
            token_lower = token.lower().strip('.,!?;:()"')
            
            if (token_lower in synonyms_dict and 
                words_replaced < max_replacements and 
                random.random() < 0.4):  # Reduced replacement probability
                
                synonyms = synonyms_dict[token_lower]
                
                # For maritime context, check if the synonym is contextually appropriate
                if has_maritime_context and token_lower in ['ship', 'boat', 'move', 'fast', 'slow', 'safe', 'dangerous']:
                    synonym = random.choice(synonyms)
                else:
                    # For general terms, prefer academic alternatives
                    synonym = random.choice(synonyms)
                
                # Preserve capitalization and punctuation
                if token[0].isupper():
                    synonym = synonym.capitalize()
                
                punctuation = ''.join(c for c in token if not c.isalnum() and c not in [' ', '-'])
                new_tokens.append(synonym + punctuation)
                words_replaced += 1
            else:
                new_tokens.append(token)
        
        return ' '.join(new_tokens)

    def enhance_terminology(self, sentence, language):
        """
        Enhanced terminology replacement with better context awareness (more conservative)
        """
        # Only enhance if sentence contains multiple maritime indicators
        maritime_words_in_sentence = sum(1 for keyword in self.maritime_keywords.get(language, []) 
                                       if keyword in sentence.lower())
        
        if maritime_words_in_sentence < 2:
            return sentence  # Don't enhance if insufficient maritime context
        
        # Conservative enhancements - only most common and contextually appropriate
        enhancements = {
            'en': {
                'navigation': 'marine navigation',
                'safety': 'operational safety',
                'communication': 'ship communication',
                'engine': 'propulsion system'
            },
            'fr': {
                'navigation': 'navigation maritime',
                'sécurité': 'sécurité opérationnelle',  
                'communication': 'communication navire',
                'moteur': 'système de propulsion'
            }
        }
        
        lang_enhancements = enhancements.get(language, {})
        
        # Only one enhancement per sentence and only with low probability
        for term, enhancement in lang_enhancements.items():
            if (re.search(r'\b' + re.escape(term) + r'\b', sentence.lower()) and 
                random.random() < 0.15):  # Very low probability
                pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
                sentence = pattern.sub(enhancement, sentence, count=1)
                break  # Only one replacement per sentence
        
        return sentence

    def _get_past_participle_en(self, verb_lemma):
        """
        Get English past participle with expanded dictionary
        """
        participles = {
            'make': 'made', 'take': 'taken', 'give': 'given', 'see': 'seen',
            'do': 'done', 'go': 'gone', 'come': 'come', 'know': 'known',
            'get': 'gotten', 'find': 'found', 'think': 'thought',
            'say': 'said', 'tell': 'told', 'use': 'used', 'show': 'shown',
            'write': 'written', 'read': 'read', 'hear': 'heard', 'feel': 'felt',
            'create': 'created', 'produce': 'produced', 'analyze': 'analyzed',
            'study': 'studied', 'examine': 'examined', 'investigate': 'investigated'
        }
        return participles.get(verb_lemma, verb_lemma + 'ed')

    def _get_past_participle_fr(self, verb_lemma):
        """
        Get French past participle with expanded dictionary
        """
        participles = {
            'faire': 'fait', 'dire': 'dit', 'voir': 'vu', 'prendre': 'pris',
            'donner': 'donné', 'mettre': 'mis', 'écrire': 'écrit',
            'lire': 'lu', 'comprendre': 'compris', 'utiliser': 'utilisé',
            'effectuer': 'effectué', 'réaliser': 'réalisé', 'analyser': 'analysé',
            'étudier': 'étudié', 'examiner': 'examiné', 'créer': 'créé'
        }
        return participles.get(verb_lemma)


# Enhanced convenience functions
def process_academic_text(text, language=None, conservative=True):
    """
    Convenience function for processing academic text with conservative settings
    """
    if conservative:
        humanizer = MaritimeAcademicTextHumanizer(
            p_passive=0.1,
            p_synonym_replacement=0.2,
            p_academic_transition=0.15,
            p_maritime_terminology=0.1,  # Very low for general academic text
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

def process_formal_text(text, language=None, formality_level='medium'):
    """
    Process text for formal writing with different formality levels
    """
    if formality_level == 'high':
        p_synonym = 0.3
        p_transition = 0.2
        p_maritime = 0.15
        p_passive = 0.12
    elif formality_level == 'medium':
        p_synonym = 0.22
        p_transition = 0.15
        p_maritime = 0.1
        p_passive = 0.08
    else:  # low
        p_synonym = 0.15
        p_transition = 0.1
        p_maritime = 0.05
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