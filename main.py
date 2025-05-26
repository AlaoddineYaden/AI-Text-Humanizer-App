import os
import sys

# CRITICAL: Set environment variables BEFORE importing any other modules
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'
os.environ['STREAMLIT_SERVER_RUN_ON_SAVE'] = 'false'

# Fix for PyTorch + Streamlit compatibility
try:
    import torch
    # Additional PyTorch compatibility fixes
    torch.set_num_threads(1)
except ImportError:
    pass

import streamlit as st
from transformer.app import MaritimeAcademicTextHumanizer, NLP_GLOBAL, download_nltk_resources
from nltk.tokenize import word_tokenize
import spacy


def get_nlp_model(language='auto'):
    if language == 'auto' and NLP_GLOBAL:
        return NLP_GLOBAL
    elif language == 'en':
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            return NLP_GLOBAL
    elif language == 'fr':
        try:
            return spacy.load("fr_core_news_sm")
        except OSError:
            return NLP_GLOBAL
    else:
        return NLP_GLOBAL


def main():
    download_nltk_resources()

    st.set_page_config(
        page_title="Maritime Text Enhancer",
        page_icon="🚢",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
        <style>
        html, body, [class*="css"]  {
            font-family: 'Segoe UI', sans-serif;
        }
        .block-container {
            padding: 2rem 2rem 2rem 2rem;
        }
        .title {
            text-align: center;
            font-size: 2.4rem;
            font-weight: 700;
            color: #fff;
        }
        .info-box {
            background-color: #f0f4f8;
            padding: 1rem;
            border-left: 5px solid #004080;
            border-radius: 5px;
            color: #000;
        }
        .stat-box {
            background-color: #e8f1fa;
            padding: 1rem;
            border-radius: 5px;
            color: #000;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='title'>Maritime Academic Text Enhancer</div>", unsafe_allow_html=True)
    st.markdown("""
        <div class='info-box'>
        Transform plain text into professionally enhanced maritime academic language. Perfect for students, researchers, and professionals in naval science.
        </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("🔧 Settings")
        language = st.selectbox("Language", ['auto', 'en', 'fr'], format_func=lambda x: {'auto': 'Auto', 'en': 'English', 'fr': 'Français'}[x])
        formality = st.radio("Formality", ['medium', 'low',  'high'], format_func=lambda x: {'medium': '📖 Standard','low': '📝 Light',  'high': '📚 High Academic'}[x])

        st.markdown("---")
        use_passive = st.toggle("Enable Passive Voice", value=False)
        use_synonyms = st.toggle("Use Academic Synonyms", value=True)
        use_maritime = st.toggle("Enhance Maritime Vocabulary", value=True)

        st.markdown("---")
        st.subheader("🎛️ Fine-tuning")
        p_synonym = st.slider("Synonym Probability", 0.0, 1.0, 0.3)
        p_transition = st.slider("Transitions Probability", 0.0, 1.0, 0.3)
        p_maritime = st.slider("Maritime Terms Probability", 0.0, 1.0, 0.4)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📝 Input Text")
        user_text = st.text_area("Paste or type your text", height=300)
        uploaded_file = st.file_uploader("Or upload a text file", type="txt")

        if uploaded_file:
            user_text = uploaded_file.read().decode("utf-8", errors="ignore")
            st.success("Text file loaded.")

    with col2:
        st.subheader("🔍 Enhanced Output")
        if st.button("🚀 Transform Text"):
            if not user_text.strip():
                st.warning("Please provide some text.")
            else:
                with st.spinner("Enhancing your text..."):
                    nlp_model = get_nlp_model(language)

                    try:
                        lang_detected = 'french' if 'le ' in user_text.lower() else 'english'
                        in_words = len(word_tokenize(user_text, language=lang_detected))
                        in_sentences = len(list(nlp_model(user_text).sents))
                    except:
                        in_words = len(user_text.split())
                        in_sentences = user_text.count('.')

                    probs = {'low': (0.2, 0.2, 0.3), 'medium': (0.3, 0.3, 0.4), 'high': (0.4, 0.4, 0.5)}
                    syn, trans, mar = probs[formality]

                    humanizer = MaritimeAcademicTextHumanizer(
                        p_passive=0.2,
                        p_synonym_replacement=p_synonym if abs(p_synonym - 0.3) > 0.05 else syn,
                        p_academic_transition=p_transition if abs(p_transition - 0.3) > 0.05 else trans,
                        p_maritime_terminology=p_maritime if abs(p_maritime - 0.4) > 0.05 else mar,
                        seed=42
                    )

                    transformed = humanizer.humanize_text(
                        user_text,
                        language=None if language == 'auto' else language,
                        use_passive=use_passive,
                        use_synonyms=use_synonyms,
                        use_maritime_terms=use_maritime
                    )

                    try:
                        out_words = len(word_tokenize(transformed, language=lang_detected))
                        out_sentences = len(list(nlp_model(transformed).sents))
                    except:
                        out_words = len(transformed.split())
                        out_sentences = transformed.count('.')


                    st.text_area("Enhanced Text", transformed, height=300)

                    st.markdown(f"""
                        <div class='stat-box'>
                        <b>📊 Stats:</b><br>
                        Words: {in_words} → {out_words} - 
                        Sentences: {in_sentences} → {out_sentences} - 
                        Change: {((out_words - in_words) / in_words * 100):+.1f}%
                        </div>
                    """, unsafe_allow_html=True)
                    

                    

    st.markdown("---")
    with st.expander("📚 Examples & Tips"):
        st.markdown("""
        ### English Example
        - **Input:** "The ship moves fast."
        - **Output:** "The maritime vessel progresses at a swift pace."

        ### French Example
        - **Entrée:** "Le navire avance."
        - **Sortie:** "Le bâtiment maritime effectue une progression."

        **Tips**:
        - Use formal phrases for best results.
        - The longer the text, the more noticeable the enhancements.
        - High formality is best for academic submissions.
        """)

    st.markdown("""
        <div style='text-align: center; color: #999; padding-top: 2rem;'>
        Maritime Text Enhancer | Created for Naval Science Professionals ⚓
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()