import streamlit as st
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import matplotlib.pyplot as plt
import spacy
from collections import Counter
import numpy as np
from textstat import flesch_reading_ease, flesch_kincaid_grade
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import pandas as pd

# Ensure necessary NLTK resources
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

# Ensure spaCy model is available
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()


# --- PAGE CONFIG ---
st.set_page_config(page_title="Smart Resume Analyzer",
                   layout="wide", page_icon="📄")

# --- SESSION STATE INIT ---
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
if 'rerun_trigger' not in st.session_state:
    st.session_state.rerun_trigger = False


def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode


# --- UTILITY FUNCTIONS ---
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    return " ".join([page.get_text() for page in doc])


def clean_text(text):
    return re.sub(r'\s+', ' ', text.strip().lower())


def compute_similarity(resume_text, jd_text):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, jd_text])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])
    return round(similarity[0][0] * 100, 2)


def extract_skills_nlp(text):
    possible_skills = [
        "python", "java", "c++", "c", "sql", "html", "css", "javascript",
        "react", "node.js", "express", "django", "flask", "machine learning",
        "deep learning", "data science", "cloud", "aws", "azure", "gcp",
        "linux", "git", "github", "tensorflow", "pytorch", "nlp", "opencv",
        "kubernetes", "docker", "mongodb", "mysql", "postgresql", "bash",
        "shell", "json", "api", "rest", "graphql", "pandas", "numpy",
        "matplotlib", "seaborn", "power bi", "tableau", "etl", "spark",
        "hadoop", "airflow", "agile", "scrum", "communication", "leadership",
        "project management", "teamwork", "problem solving", "analytics"
    ]
    text = clean_text(text)
    return {skill for skill in possible_skills if skill in text}


def extract_keywords_spacy(text, n=20):
    doc = nlp(text)
    words = [token.text.lower()
             for token in doc if token.is_alpha and not token.is_stop]
    most_common = Counter(words).most_common(n)
    return {word for word, _ in most_common}


def analyze_resume_structure(text):
    """Analyze resume structure and content quality"""
    doc = nlp(text)

    # Basic metrics
    word_count = len([token for token in doc if token.is_alpha])
    sentence_count = len(list(doc.sents))
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0

    # Check for key sections
    sections = {
        'experience': any(keyword in text.lower() for keyword in ['experience', 'work history', 'employment', 'career']),
        'education': any(keyword in text.lower() for keyword in ['education', 'degree', 'university', 'college', 'school']),
        'skills': any(keyword in text.lower() for keyword in ['skills', 'competencies', 'technologies']),
        'projects': any(keyword in text.lower() for keyword in ['projects', 'portfolio', 'achievements']),
        'contact': any(keyword in text.lower() for keyword in ['email', 'phone', 'contact', '@', 'linkedin'])
    }

    # Action verbs analysis
    action_verbs = [
        'managed', 'led', 'developed', 'created', 'implemented', 'designed',
        'improved', 'optimized', 'analyzed', 'collaborated', 'achieved',
        'delivered', 'executed', 'coordinated', 'supervised', 'trained'
    ]
    action_verb_count = sum(1 for verb in action_verbs if verb in text.lower())

    # Quantifiable achievements
    numbers_pattern = r'\b\d+(?:\.\d+)?(?:%|k|million|billion|years?|months?)\b'
    quantifiable_achievements = len(re.findall(numbers_pattern, text.lower()))

    return {
        'word_count': word_count,
        'sentence_count': sentence_count,
        'avg_sentence_length': avg_sentence_length,
        'sections': sections,
        'action_verb_count': action_verb_count,
        'quantifiable_achievements': quantifiable_achievements,
        'readability_score': flesch_reading_ease(text) if text else 0
    }


def generate_ml_suggestions(resume_analysis, resume_skills, jd_skills, missing_skills, similarity_score):
    """Generate improvement suggestions based on ML/NLP analysis"""
    suggestions = []

    # 1. Skills Gap Analysis
    if missing_skills:
        top_missing = list(missing_skills)[:5]
        suggestions.append({
            'category': 'Skills Enhancement',
            'priority': 'High',
            'suggestion': f"Add these in-demand skills to strengthen your profile: {', '.join(top_missing)}",
            'impact': 'High - Increases keyword matching by 15-25%'
        })

    # 2. Content Length Analysis
    if resume_analysis['word_count'] < 200:
        suggestions.append({
            'category': 'Content Depth',
            'priority': 'High',
            'suggestion': "Your resume is too brief. Expand descriptions to 300-600 words for better impact.",
            'impact': 'High - Improves content richness and keyword density'
        })
    elif resume_analysis['word_count'] > 800:
        suggestions.append({
            'category': 'Content Conciseness',
            'priority': 'Medium',
            'suggestion': "Consider condensing content. Aim for 400-600 words for optimal readability.",
            'impact': 'Medium - Improves recruiter attention span'
        })

    # 3. Action Verbs Analysis
    if resume_analysis['action_verb_count'] < 5:
        suggestions.append({
            'category': 'Language Strength',
            'priority': 'High',
            'suggestion': "Use more action verbs (managed, developed, implemented, optimized) to show impact.",
            'impact': 'High - Makes achievements more compelling'
        })

    # 4. Quantifiable Achievements
    if resume_analysis['quantifiable_achievements'] < 3:
        suggestions.append({
            'category': 'Achievement Quantification',
            'priority': 'High',
            'suggestion': "Add specific numbers, percentages, or metrics to demonstrate your impact.",
            'impact': 'Very High - Quantified results are 40% more effective'
        })

    # 5. Section Completeness
    missing_sections = [section for section,
                        present in resume_analysis['sections'].items() if not present]
    if missing_sections:
        suggestions.append({
            'category': 'Structure Completeness',
            'priority': 'Medium',
            'suggestion': f"Consider adding missing sections: {', '.join(missing_sections).title()}",
            'impact': 'Medium - Improves professional presentation'
        })

    # 6. Readability Analysis
    if resume_analysis['readability_score'] < 60:
        suggestions.append({
            'category': 'Readability',
            'priority': 'Medium',
            'suggestion': "Simplify sentence structure for better readability (current score: {:.1f}/100)".format(resume_analysis['readability_score']),
            'impact': 'Medium - Easier for ATS and recruiters to parse'
        })

    # 7. Similarity Score Based Suggestions
    if similarity_score < 50:
        suggestions.append({
            'category': 'Job Relevance',
            'priority': 'High',
            'suggestion': "Tailor your resume more closely to the job description. Current match: {:.1f}%".format(similarity_score),
            'impact': 'Very High - Improves ATS ranking significantly'
        })

    # 8. Skills Diversity
    if len(resume_skills) < 8:
        suggestions.append({
            'category': 'Skills Breadth',
            'priority': 'Medium',
            'suggestion': "Expand your skills section to include both technical and soft skills.",
            'impact': 'Medium - Shows well-rounded capabilities'
        })

    return suggestions


def calculate_resume_score(resume_analysis, similarity_score, matched_skills, missing_skills):
    """Calculate overall resume score based on multiple factors"""
    score = 0
    max_score = 100

    # Similarity score (30% weight)
    score += (similarity_score / 100) * 30

    # Skills match (25% weight)
    total_relevant_skills = len(matched_skills) + len(missing_skills)
    if total_relevant_skills > 0:
        skill_match_ratio = len(matched_skills) / total_relevant_skills
        score += skill_match_ratio * 25

    # Content quality (20% weight)
    content_score = min(resume_analysis['word_count'] / 500, 1) * 10
    content_score += min(resume_analysis['action_verb_count'] / 8, 1) * 5
    content_score += min(
        resume_analysis['quantifiable_achievements'] / 5, 1) * 5
    score += content_score

    # Structure completeness (15% weight)
    sections_present = sum(resume_analysis['sections'].values())
    score += (sections_present / 5) * 15

    # Readability (10% weight)
    readability_normalized = min(resume_analysis['readability_score'] / 100, 1)
    score += readability_normalized * 10

    return min(score, max_score)


# --- THEME STYLING ---
if st.session_state.dark_mode:
    bg_color = "#2E0854"
    text_color = "#e0e0e0"
    btn_bg = "#5a33b0"
    btn_color = "#e0e0e0"
else:
    bg_color = "#ADD8E6"
    text_color = "#ffffff"
    btn_bg = "#4CAF50"
    btn_color = "#ffffff"

st.markdown(
    f"""
    <style>
        .css-18e3th9 {{
            background-color: {bg_color} !important;
            color: {text_color} !important;
        }}
        .main {{
            background-color: transparent;
            color: {text_color};
        }}
        .stButton>button {{
            background-color: {btn_bg};
            color: {btn_color};
            padding: 0.5em 2em;
            border: none;
            border-radius: 10px;
            font-weight: bold;
            font-size: 1em;
        }}
        .stButton>button:hover {{
            background-color: #45a049;
            transition: 0.3s;
        }}
        .stFileUploader label {{
            font-weight: bold;
            color: {text_color};
        }}
        .stExpanderHeader {{
            color: {text_color} !important;
        }}
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}
        ::-webkit-scrollbar-track {{
            background: {bg_color};
        }}
        ::-webkit-scrollbar-thumb {{
            background-color: {btn_bg};
            border-radius: 10px;
            border: 2px solid {bg_color};
        }}
    </style>
    """, unsafe_allow_html=True
)

# --- HEADER ---
st.markdown(f"""
    <div style='text-align: center;'>
        <h1 style='color: {btn_bg};'>📄 Smart Resume Analyzer</h1>
        <p style='color: gray;'>Match your resume with job descriptions using Advanced NLP & ML</p>
    </div>
""", unsafe_allow_html=True)

# --- THEME TOGGLE BUTTON ---
st.sidebar.button(
    "Fun Toggle 🌙" if not st.session_state.dark_mode else "Fun Toggle ☀️",
    on_click=toggle_theme
)

# --- FILE UPLOAD ---
st.markdown("### 📤 Upload Files")
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        resume_file = st.file_uploader("🎓 Upload Resume (PDF)", type=["pdf"])
    with col2:
        jd_file = st.file_uploader(
            "💼 Upload Job Description (PDF or TXT)", type=["pdf", "txt"])

# --- INITIALIZE VARIABLES ---
resume_skills = set()
jd_skills = set()
matched_skills = set()
missing_skills = set()
score = 0
resume_text = ""
resume_analysis = None

if resume_file and jd_file:
    with st.spinner("Analyzing your resume using Advanced ML & NLP..."):
        resume_text = extract_text_from_pdf(resume_file)
        jd_text = (
            extract_text_from_pdf(jd_file)
            if jd_file.name.endswith(".pdf")
            else jd_file.read().decode("utf-8", errors="ignore")
        )

        # Clean texts
        clean_resume = clean_text(resume_text)
        clean_jd = clean_text(jd_text)

        # Compute similarity and extract skills
        score = compute_similarity(clean_resume, clean_jd)
        resume_skills = extract_skills_nlp(clean_resume)
        jd_skills = extract_keywords_spacy(clean_jd, n=25)
        matched_skills = resume_skills & jd_skills
        missing_skills = jd_skills - resume_skills

        # Analyze resume structure
        resume_analysis = analyze_resume_structure(resume_text)

# --- MATCH SCORE DISPLAY ---
st.markdown("### 🎯 Resume Match Score")
st.progress(score / 100)
if score > 0:
    st.success(f"Your Resume matches the Job Description by **{score}%**")
else:
    st.info("Upload both Resume and Job Description to see the match score.")

# --- OVERALL RESUME SCORE ---
if resume_analysis:
    overall_score = calculate_resume_score(
        resume_analysis, score, matched_skills, missing_skills)
    st.markdown("### 📊 Overall Resume Quality Score")
    st.progress(overall_score / 100)

    if overall_score >= 80:
        st.success(
            f"Excellent! Your resume scores **{overall_score:.1f}/100**")
    elif overall_score >= 60:
        st.warning(
            f"Good! Your resume scores **{overall_score:.1f}/100** - Room for improvement")
    else:
        st.error(
            f"Your resume scores **{overall_score:.1f}/100** - Needs significant improvement")

# --- SKILLS ANALYSIS SECTION ---
st.markdown("### 🧠 Skills Analysis")

with st.expander("🔍 Detailed Skills Report", expanded=True):
    st.markdown(
        f"✅ **Matched Skills:** `{', '.join(sorted(matched_skills)) or 'None'}`")
    st.markdown(
        f"❌ **Missing Skills:** `{', '.join(sorted(missing_skills)) or 'None'}`")

# --- CHARTS SECTION ---
if matched_skills or missing_skills:
    st.markdown("#### Skills Match Visualizations")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Pie Chart**")
        fig, ax = plt.subplots(figsize=(4, 4))
        matched_count = len(matched_skills)
        missing_count = len(missing_skills)
        labels = ['Matched Skills', 'Missing Skills']
        sizes = [matched_count, missing_count]
        colors = ['#00C49F', '#FF4C61']
        ax.pie(sizes, labels=labels, colors=colors,
               autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)

    with col2:
        st.markdown("**Bar Chart**")
        fig, ax = plt.subplots(figsize=(4, 4))
        counts = [matched_count, missing_count]
        bars = ax.bar(['Matched Skills', 'Missing Skills'],
                      counts, color=colors)
        ax.set_ylabel('Count')
        ax.set_title('Skills Match Bar Chart')
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.1,
                    int(yval), ha='center', va='bottom')
        st.pyplot(fig)

    with col3:
        st.markdown("**Radar Chart**")
        labels = list(jd_skills)
        if not labels:
            st.info("Not enough skills to display radar chart.")
        else:
            num_vars = len(labels)
            angles = np.linspace(0, 2 * np.pi, num_vars,
                                 endpoint=False).tolist()
            labels += labels[:1]
            angles += angles[:1]

            matched_vals = [
                1 if skill in matched_skills else 0 for skill in jd_skills]
            matched_vals += matched_vals[:1]

            fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
            ax.plot(angles, matched_vals, color="#00C49F",
                    linewidth=2, label='Matched Skills')
            ax.fill(angles, matched_vals, color="#00C49F", alpha=0.25)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels[:-1], fontsize=8)
            ax.set_yticks([0, 0.5, 1])
            ax.set_yticklabels(['0', '', '1'])
            ax.set_ylim(0, 1)
            ax.set_title("Skill Coverage Radar Chart")
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            st.pyplot(fig)

# --- ML/NLP BASED SUGGESTIONS ---
st.markdown("### 🤖 AI-Powered Resume Improvement Suggestions")

if resume_file and resume_analysis:
    suggestions = generate_ml_suggestions(
        resume_analysis, resume_skills, jd_skills, missing_skills, score)

    if suggestions:
        st.markdown("#### 📋 Personalized Recommendations")

        # Group suggestions by priority
        high_priority = [s for s in suggestions if s['priority'] == 'High']
        medium_priority = [s for s in suggestions if s['priority'] == 'Medium']

        if high_priority:
            st.markdown("##### 🔥 High Priority Improvements")
            for i, suggestion in enumerate(high_priority, 1):
                with st.expander(f"{i}. {suggestion['category']}", expanded=True):
                    st.markdown(f"**Suggestion:** {suggestion['suggestion']}")
                    st.markdown(f"**Expected Impact:** {suggestion['impact']}")

        if medium_priority:
            st.markdown("##### 📈 Medium Priority Improvements")
            for i, suggestion in enumerate(medium_priority, 1):
                with st.expander(f"{i}. {suggestion['category']}", expanded=False):
                    st.markdown(f"**Suggestion:** {suggestion['suggestion']}")
                    st.markdown(f"**Expected Impact:** {suggestion['impact']}")

    # --- DETAILED ANALYTICS ---
    with st.expander("📊 Detailed Resume Analytics", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Content Metrics:**")
            st.write(f"• Word Count: {resume_analysis['word_count']}")
            st.write(f"• Sentences: {resume_analysis['sentence_count']}")
            st.write(
                f"• Avg Sentence Length: {resume_analysis['avg_sentence_length']:.1f}")
            st.write(
                f"• Action Verbs Used: {resume_analysis['action_verb_count']}")
            st.write(
                f"• Quantified Achievements: {resume_analysis['quantifiable_achievements']}")

        with col2:
            st.markdown("**Section Analysis:**")
            for section, present in resume_analysis['sections'].items():
                status = "✅" if present else "❌"
                st.write(f"• {section.title()}: {status}")
            st.write(
                f"• Readability Score: {resume_analysis['readability_score']:.1f}/100")

else:
    st.info("📤 Upload a resume to get AI-powered improvement suggestions based on advanced NLP and ML analysis.")

# --- FOOTER ---
st.markdown(f"""
---
<div style='text-align: center; color: gray;'>
    <p>Built by Team-13: Suman Mahapatra, Satyam Singh Chandel</p>
</div>
""", unsafe_allow_html=True)
