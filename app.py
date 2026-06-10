# ═══════════════════════════════════════════════════════════════
#  AI-POWERED MULTIMODAL RESUME BUILDER
#  Idea 2 — Chat → GenAI builds resume → BiLSTM verifies →
#            Score → PDF export
#
#  Stack:
#    Conversation    → LangChain + Streamlit chat
#    Resume content  → Google Gemini 2.5 Flash
#    Classification  → BiLSTM + Custom Attention
#    PDF generation  → ReportLab
#    Optional input  → pdfplumber (existing resume upload)
# ═══════════════════════════════════════════════════════════════
# -*- coding: utf-8 -*-
import os
import re
import json
import time
import pickle
import warnings
import textwrap
import numpy as np
import streamlit as st
import pdfplumber
import google.generativeai as genai
import tensorflow as tf
from io import BytesIO
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ReportLab imports for PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

# LangChain imports

from langchain_community.chat_message_histories import StreamlitChatMessageHistory

warnings.filterwarnings("ignore")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


# ══════════════════════════════════════════════════════════════
#  1. GEMINI + LANGCHAIN SETUP
# ══════════════════════════════════════════════════════════════


def get_api_key() -> str:
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.getenv("GEMINI_API_KEY", "")


def setup_gemini():
    """Setup raw Gemini model for resume generation and scoring."""
    api_key = get_api_key()
    if not api_key:
        st.error(
            "❌ Gemini API key missing.\n\n"
            "**Streamlit Cloud:** Settings → Secrets → add:\n"
            "```\nGEMINI_API_KEY = 'your_key'\n```\n"
            "Get free key: https://aistudio.google.com"
        )
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")


def chat_with_memory(gemini_model, user_message: str,
                     chat_history: list) -> str:
    """
    Send a message to Gemini with full conversation history.

    How memory works WITHOUT LangChain:
    - We store every message in st.session_state.chat_history
    - Every API call includes the FULL history as context
    - Gemini reads all previous Q&A and responds accordingly
    - This is identical to what LangChain was doing internally

    Parameters:
        gemini_model : Gemini GenerativeModel instance
        user_message : the current user input string
        chat_history : list of {role, content} dicts — full history

    Returns:
        str — Gemini's response
    """

    # Build conversation history as a single context string
    # Format: "User: ... \nAssistant: ... \nUser: ..."
    history_text = ""
    for msg in chat_history:
        role    = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"]
        history_text += f"{role}: {content}\n\n"

    # Full prompt = system instruction + history + new message
    full_prompt = f"""You are a professional resume writing assistant.
Your job is to collect information from the user through a friendly
conversation to build their resume.

Ask ONE question at a time. Be warm, encouraging, and professional.
When the user answers, acknowledge their answer briefly and ask
the next question.

Previous conversation:
{history_text}

User: {user_message}


# ══════════════════════════════════════════════════════════════
#  2. ATTENTION LAYER — must match training exactly
# ══════════════════════════════════════════════════════════════


class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="attention_W",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="attention_b", shape=(1,), initializer="zeros", trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        e = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        context = tf.reduce_sum(x * a, axis=1)
        return context

    def get_config(self):
        return super().get_config()


# ══════════════════════════════════════════════════════════════
#  3. MODEL BUILDER
# ══════════════════════════════════════════════════════════════


def build_model(num_classes=43, max_vocab=25000, max_len=300):
    l2 = tf.keras.regularizers.l2(1e-4)

    inputs = tf.keras.Input(shape=(max_len,), name="resume_input")
    x = layers.Embedding(
        input_dim=max_vocab,
        output_dim=256,
        embeddings_regularizer=l2,
        name="word_embedding",
    )(inputs)
    x = layers.SpatialDropout1D(0.2, name="spatial_dropout")(x)
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.1),
        name="bilstm",
    )(x)

    attn_out = AttentionLayer(name="attention")(x)
    maxpool_out = layers.GlobalMaxPooling1D(name="global_max_pool")(x)
    x = layers.Concatenate(name="combine")([attn_out, maxpool_out])

    x = layers.BatchNormalization(name="batch_norm_1")(x)
    x = layers.Dropout(0.4, name="dropout_1")(x)
    x = layers.Dense(512, activation="relu", kernel_regularizer=l2, name="dense_1")(x)
    x = layers.BatchNormalization(name="batch_norm_2")(x)
    x = layers.Dropout(0.35, name="dropout_2")(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=l2, name="dense_2")(x)
    x = layers.BatchNormalization(name="batch_norm_3")(x)
    x = layers.Dropout(0.25, name="dropout_3")(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=l2, name="dense_3")(x)
    x = layers.Dropout(0.2, name="dropout_4")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="ResumeClassifier")


# ══════════════════════════════════════════════════════════════
#  4. LOAD ARTEFACTS
# ══════════════════════════════════════════════════════════════


@st.cache_resource
def load_classifier():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    assets = os.path.join(curr_dir, "streamlit_assets")

    weights_path = os.path.join(assets, "model_weights.weights.h5")
    tokenizer_path = os.path.join(assets, "tokenizer.pkl")
    le_path = os.path.join(assets, "label_encoder.pkl")
    config_path = os.path.join(assets, "config.json")

    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        max_len = cfg.get("MAX_LEN", 300)
        max_vocab = cfg.get("MAX_VOCAB", 25000)
        num_classes = cfg.get("NUM_CLASSES", 43)
    else:
        max_len, max_vocab, num_classes = 300, 25000, 43

    model = build_model(num_classes=num_classes, max_vocab=max_vocab, max_len=max_len)
    _ = model(tf.zeros((1, max_len), dtype=tf.int32), training=False)

    if not os.path.exists(weights_path):
        st.error(f"❌ Weights not found: {weights_path}")
        return None, None, None, None

    try:
        model.load_weights(weights_path)
    except Exception as e:
        st.error(f"❌ Weight loading failed:\n{e}")
        return None, None, None, None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)
        with open(le_path, "rb") as f:
            le = pickle.load(f)

    return model, tokenizer, le, max_len


# ══════════════════════════════════════════════════════════════
#  5. NLTK STOPWORDS — with retry and fallback
# ══════════════════════════════════════════════════════════════

import nltk


def _download_stopwords(max_attempts=3):
    for attempt in range(max_attempts):
        try:
            nltk.download("stopwords", quiet=True, force=True)
            from nltk.corpus import stopwords as _sw

            _ = _sw.words("english")
            return True
        except Exception:
            if attempt < max_attempts - 1:
                time.sleep(2)
    return False


_nltk_ok = _download_stopwords()

if _nltk_ok:
    from nltk.corpus import stopwords as _sw

    _STOPS = set(_sw.words("english"))
else:
    _STOPS = {
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "he",
        "him",
        "his",
        "she",
        "her",
        "they",
        "them",
        "what",
        "this",
        "that",
        "am",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "a",
        "an",
        "the",
        "and",
        "but",
        "or",
        "as",
        "of",
        "at",
        "by",
        "for",
        "with",
        "in",
        "out",
        "on",
        "to",
        "from",
        "not",
    }

RESUME_GENERIC = {
    "experience",
    "work",
    "company",
    "team",
    "management",
    "skills",
    "responsibilities",
    "working",
    "worked",
    "years",
    "role",
    "position",
    "job",
    "career",
    "professional",
    "strong",
    "ability",
    "knowledge",
    "excellent",
    "good",
    "using",
    "used",
    "use",
    "provide",
    "support",
    "responsible",
    "manage",
    "develop",
    "maintain",
}
ALL_STOP = _STOPS.union(RESUME_GENERIC)


def clean_text(text: str) -> str:
    text = re.sub(r"http\S+|www\S+|\S+@\S+|<.*?>", " ", str(text))
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).lower().strip()
    return " ".join(w for w in text.split() if w not in ALL_STOP and len(w) > 2)


# ══════════════════════════════════════════════════════════════
#  6. CLASSIFY RESUME TEXT — BiLSTM + Attention
# ══════════════════════════════════════════════════════════════


def classify_text(text, model, tokenizer, le, max_len, top_n=3):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    proba = model.predict(padded, verbose=0)[0]
    top_idx = np.argsort(proba)[::-1][:top_n]
    return [
        {"category": le.classes_[i], "confidence": float(proba[i])} for i in top_idx
    ]


# ══════════════════════════════════════════════════════════════
#  7. THE 10 CHAT QUESTIONS
#  LangChain bot asks these one by one.
#  Each question is sent as a message to the ConversationChain.
# ══════════════════════════════════════════════════════════════

QUESTIONS = [
    "What is your full name?",
    "What is your contact information? (email, phone, LinkedIn, location)",
    "What is your target job role or the position you are applying for?",
    "Write a brief professional summary about yourself (2-3 sentences about your background and goals).",
    "List your work experience. For each job include: Job Title, Company Name, Duration, and 2-3 key achievements.",
    "What is your educational background? Include degree, institution, year, and any relevant coursework.",
    "List your technical skills, tools, and technologies you are proficient in.",
    "Describe any notable projects you have worked on. Include: Project name, what it does, and technologies used.",
    "List any certifications, awards, or achievements you are proud of.",
    "Paste the job description you are targeting (or describe the role in detail so we can tailor your resume).",
]

QUESTION_KEYS = [
    "full_name",
    "contact_info",
    "target_role",
    "summary",
    "experience",
    "education",
    "skills",
    "projects",
    "certifications",
    "job_description",
]


# ══════════════════════════════════════════════════════════════
#  8. GEMINI — GENERATE FULL RESUME CONTENT
#  Takes all 10 answers → returns structured resume JSON
# ══════════════════════════════════════════════════════════════


def generate_resume_content(gemini_model, answers: dict) -> dict:
    """
    Send all collected answers to Gemini.
    Gemini writes a professional resume in structured JSON.

    Parameters:
        gemini_model : Gemini GenerativeModel
        answers      : dict with all 10 user answers

    Returns:
        dict with structured resume sections
    """
    prompt = f"""
You are an expert resume writer with 15+ years of experience.

Using the information below, write a polished, professional resume.
Tailor the content specifically for the target role.
Use strong action verbs. Quantify achievements where possible.
Make it ATS-friendly.

USER INFORMATION:
Full Name        : {answers.get('full_name', '')}
Contact Info     : {answers.get('contact_info', '')}
Target Role      : {answers.get('target_role', '')}
Professional Summary: {answers.get('summary', '')}
Work Experience  : {answers.get('experience', '')}
Education        : {answers.get('education', '')}
Technical Skills : {answers.get('skills', '')}
Projects         : {answers.get('projects', '')}
Certifications   : {answers.get('certifications', '')}
Job Description  : {answers.get('job_description', '')}

Return ONLY a valid JSON object with this exact structure:
{{
  "name": "Full name",
  "email": "email address",
  "phone": "phone number",
  "location": "city, country",
  "linkedin": "linkedin URL or empty string",
  "github": "github URL or empty string",
  "target_role": "Job title they are applying for",
  "summary": "3-4 sentence professional summary tailored to target role",
  "experience": [
    {{
      "title": "Job Title",
      "company": "Company Name",
      "duration": "Start – End (e.g. Jan 2022 – Present)",
      "bullets": [
        "Achievement 1 with action verb and metrics",
        "Achievement 2 with action verb and metrics",
        "Achievement 3 with action verb and metrics"
      ]
    }}
  ],
  "education": [
    {{
      "degree": "Degree name",
      "institution": "University/College Name",
      "year": "Graduation year",
      "details": "Relevant coursework or GPA if notable"
    }}
  ],
  "skills": {{
    "technical": ["skill1", "skill2", "skill3"],
    "tools": ["tool1", "tool2", "tool3"],
    "soft": ["skill1", "skill2"]
  }},
  "projects": [
    {{
      "name": "Project Name",
      "description": "1-2 line description with impact",
      "tech": ["tech1", "tech2"]
    }}
  ],
  "certifications": ["cert1", "cert2"],
  "linkedin_summary": "A compelling 3-sentence LinkedIn About section"
}}
"""

    for attempt in range(3):
        try:
            response = gemini_model.generate_content(prompt)
            raw = response.text.strip()
            raw = re.sub(r"```json\s*", "", raw)
            raw = re.sub(r"```\s*", "", raw)
            return json.loads(raw.strip())

        except json.JSONDecodeError:
            if attempt < 2:
                time.sleep(3)
            else:
                st.error("❌ Could not parse resume content. Please try again.")
                return {}
        except Exception as e:
            if "429" in str(e):
                wait = 60 * (attempt + 1)
                st.warning(f"⏳ Rate limit. Waiting {wait}s...")
                time.sleep(wait)
            else:
                st.error(f"❌ Gemini error: {e}")
                return {}
    return {}


# ══════════════════════════════════════════════════════════════
#  9. GEMINI — SCORE RESUME ON 10 CRITERIA
# ══════════════════════════════════════════════════════════════


def score_resume(gemini_model, resume_data: dict, predicted_category: str) -> dict:
    """
    Score the generated resume on 10 professional criteria.
    Returns scores and ATS keywords.
    """
    resume_text = json.dumps(resume_data, indent=2)

    prompt = f"""
You are a senior HR professional and resume coach.

Score this resume for a {predicted_category} role on 10 criteria.
Each criterion is scored out of 10.

RESUME:
{resume_text[:2500]}

Return ONLY valid JSON:
{{
  "scores": {{
    "professional_summary":    <0-10>,
    "work_experience_quality": <0-10>,
    "skills_relevance":        <0-10>,
    "achievements_impact":     <0-10>,
    "education_presentation":  <0-10>,
    "ats_friendliness":        <0-10>,
    "action_verbs_usage":      <0-10>,
    "quantified_results":      <0-10>,
    "overall_formatting":      <0-10>,
    "job_title_alignment":     <0-10>
  }},
  "total": <sum out of 100>,
  "grade": "<A+/A/B+/B/C/D>",
  "ats_keywords": ["keyword1","keyword2","keyword3","keyword4","keyword5"],
  "missing_keywords": ["keyword1","keyword2","keyword3"],
  "top_improvements": [
    "Improvement 1",
    "Improvement 2",
    "Improvement 3"
  ]
}}
"""

    for attempt in range(3):
        try:
            response = gemini_model.generate_content(prompt)
            raw = response.text.strip()
            raw = re.sub(r"```json\s*", "", raw)
            raw = re.sub(r"```\s*", "", raw)
            return json.loads(raw.strip())

        except json.JSONDecodeError:
            if attempt < 2:
                time.sleep(3)
            else:
                return {}
        except Exception as e:
            if "429" in str(e):
                time.sleep(60 * (attempt + 1))
            else:
                return {}
    return {}


# ══════════════════════════════════════════════════════════════
#  10. REPORTLAB — GENERATE PROFESSIONAL PDF
#  Builds a beautiful single-column resume PDF from the
#  structured resume_data dictionary.
# ══════════════════════════════════════════════════════════════


def generate_pdf(resume_data: dict) -> bytes:
    """
    Generate a professional PDF resume using ReportLab.

    ReportLab builds PDFs from a 'story' -- a list of elements
    (Paragraphs, Tables, Spacers, Lines) stacked top to bottom.
    Each element is a 'Flowable' that ReportLab positions automatically.

    Parameters:
        resume_data : dict with structured resume sections

    Returns:
        bytes -- the PDF file as bytes (for st.download_button)
    """
    buffer = BytesIO()  # write to memory, not disk

    # ── Document setup ──────────────────────────────────────
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        leftMargin=2.0 * cm,
        rightMargin=2.0 * cm,
    )

    # ── Colour palette ──────────────────────────────────────
    PRIMARY = colors.HexColor("#1a1a2e")  # dark navy
    ACCENT = colors.HexColor("#0f3460")  # medium blue
    LIGHT = colors.HexColor("#e94560")  # red accent
    GRAY = colors.HexColor("#555555")
    LIGHTGRAY = colors.HexColor("#f5f5f5")

    # ── Style definitions ───────────────────────────────────
    styles = getSampleStyleSheet()

    name_style = ParagraphStyle(
        "NameStyle",
        fontName="Helvetica-Bold",
        fontSize=22,
        textColor=PRIMARY,
        alignment=TA_CENTER,
        spaceAfter=4,
    )
    contact_style = ParagraphStyle(
        "ContactStyle",
        fontName="Helvetica",
        fontSize=9,
        textColor=GRAY,
        alignment=TA_CENTER,
        spaceAfter=2,
    )
    section_style = ParagraphStyle(
        "SectionStyle",
        fontName="Helvetica-Bold",
        fontSize=11,
        textColor=ACCENT,
        spaceBefore=10,
        spaceAfter=4,
        borderPad=2,
    )
    job_title_style = ParagraphStyle(
        "JobTitleStyle",
        fontName="Helvetica-Bold",
        fontSize=10,
        textColor=PRIMARY,
        spaceAfter=1,
    )
    company_style = ParagraphStyle(
        "CompanyStyle",
        fontName="Helvetica-Oblique",
        fontSize=9,
        textColor=GRAY,
        spaceAfter=2,
    )
    bullet_style = ParagraphStyle(
        "BulletStyle",
        fontName="Helvetica",
        fontSize=9,
        textColor=colors.black,
        leftIndent=12,
        spaceAfter=2,
        leading=13,
    )
    normal_style = ParagraphStyle(
        "NormalStyle",
        fontName="Helvetica",
        fontSize=9,
        textColor=colors.black,
        spaceAfter=3,
        leading=13,
        alignment=TA_JUSTIFY,
    )
    summary_style = ParagraphStyle(
        "SummaryStyle",
        fontName="Helvetica",
        fontSize=9.5,
        textColor=colors.black,
        spaceAfter=6,
        leading=14,
        alignment=TA_JUSTIFY,
    )

    # ── Build story (list of flowable elements) ─────────────
    story = []

    # Header — Name
    name = resume_data.get("name", "Your Name")
    story.append(Paragraph(name, name_style))

    # Header — Target role
    target = resume_data.get("target_role", "")
    if target:
        role_style = ParagraphStyle(
            "RoleStyle",
            fontName="Helvetica-Bold",
            fontSize=11,
            textColor=LIGHT,
            alignment=TA_CENTER,
            spaceAfter=4,
        )
        story.append(Paragraph(target, role_style))

    # Header — Contact info
    contact_parts = []
    for key, label in [
        ("email", ""),
        ("phone", ""),
        ("location", ""),
        ("linkedin", "LinkedIn"),
        ("github", "GitHub"),
    ]:
        val = resume_data.get(key, "")
        if val:
            contact_parts.append(val)
    story.append(Paragraph("  |  ".join(contact_parts), contact_style))

    # Divider
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=2, color=ACCENT, spaceAfter=6))

    # Summary
    summary = resume_data.get("summary", "")
    if summary:
        story.append(Paragraph("PROFESSIONAL SUMMARY", section_style))
        story.append(
            HRFlowable(width="100%", thickness=0.5, color=LIGHTGRAY, spaceAfter=4)
        )
        story.append(Paragraph(summary, summary_style))

    # Experience
    experience = resume_data.get("experience", [])
    if experience:
        story.append(Paragraph("WORK EXPERIENCE", section_style))
        story.append(
            HRFlowable(width="100%", thickness=0.5, color=LIGHTGRAY, spaceAfter=4)
        )
        for exp in experience:
            # Title + Duration on same line using a table
            title = exp.get("title", "")
            company = exp.get("company", "")
            duration = exp.get("duration", "")

            dur_style = ParagraphStyle(
                "DurStyle",
                fontName="Helvetica",
                fontSize=9,
                textColor=GRAY,
                alignment=TA_LEFT,
            )
            title_para = Paragraph(f"<b>{title}</b>", job_title_style)
            duration_para = Paragraph(duration, dur_style)

            t = Table([[title_para, duration_para]], colWidths=["70%", "30%"])
            t.setStyle(
                TableStyle(
                    [
                        ("ALIGN", (0, 0), (0, 0), "LEFT"),
                        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ]
                )
            )
            story.append(t)
            story.append(Paragraph(company, company_style))

            for bullet in exp.get("bullets", []):
                story.append(Paragraph(f"• {bullet}", bullet_style))
            story.append(Spacer(1, 6))

    # Education
    education = resume_data.get("education", [])
    if education:
        story.append(Paragraph("EDUCATION", section_style))
        story.append(
            HRFlowable(width="100%", thickness=0.5, color=LIGHTGRAY, spaceAfter=4)
        )
        for edu in education:
            degree = edu.get("degree", "")
            inst = edu.get("institution", "")
            year = edu.get("year", "")
            details = edu.get("details", "")

            deg_style = ParagraphStyle(
                "DegStyle", fontName="Helvetica-Bold", fontSize=9.5, textColor=PRIMARY
            )
            yr_style = ParagraphStyle(
                "YrStyle",
                fontName="Helvetica",
                fontSize=9,
                textColor=GRAY,
                alignment=TA_LEFT,
            )
            deg_para = Paragraph(degree, deg_style)
            yr_para = Paragraph(year, yr_style)

            t = Table([[deg_para, yr_para]], colWidths=["75%", "25%"])
            t.setStyle(
                TableStyle(
                    [
                        ("ALIGN", (0, 0), (0, 0), "LEFT"),
                        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ]
                )
            )
            story.append(t)
            story.append(Paragraph(inst, company_style))
            if details:
                story.append(Paragraph(details, normal_style))
            story.append(Spacer(1, 4))

    # Skills
    skills = resume_data.get("skills", {})
    if skills:
        story.append(Paragraph("SKILLS", section_style))
        story.append(
            HRFlowable(width="100%", thickness=0.5, color=LIGHTGRAY, spaceAfter=4)
        )

        skill_label = ParagraphStyle(
            "SkillLabel", fontName="Helvetica-Bold", fontSize=9, textColor=ACCENT
        )
        for label, key in [
            ("Technical", "technical"),
            ("Tools & Technologies", "tools"),
            ("Soft Skills", "soft"),
        ]:
            items = skills.get(key, [])
            if items:
                story.append(
                    Paragraph(f"<b>{label}:</b>  {',  '.join(items)}", normal_style)
                )

    # Projects
    projects = resume_data.get("projects", [])
    if projects:
        story.append(Paragraph("PROJECTS", section_style))
        story.append(
            HRFlowable(width="100%", thickness=0.5, color=LIGHTGRAY, spaceAfter=4)
        )
        for proj in projects:
            proj_name = proj.get("name", "")
            proj_desc = proj.get("description", "")
            proj_tech = proj.get("tech", [])
            tech_str = " | ".join(proj_tech) if proj_tech else ""

            story.append(Paragraph(f"<b>{proj_name}</b>  —  {proj_desc}", normal_style))
            if tech_str:
                story.append(
                    Paragraph(
                        f"<i>Tech: {tech_str}</i>",
                        ParagraphStyle(
                            "TechStyle",
                            fontName="Helvetica-Oblique",
                            fontSize=8.5,
                            textColor=GRAY,
                            spaceAfter=4,
                        ),
                    )
                )

    # Certifications
    certs = resume_data.get("certifications", [])
    if certs:
        story.append(Paragraph("CERTIFICATIONS", section_style))
        story.append(
            HRFlowable(width="100%", thickness=0.5, color=LIGHTGRAY, spaceAfter=4)
        )
        for cert in certs:
            story.append(Paragraph(f"• {cert}", bullet_style))

    # Build the PDF
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ══════════════════════════════════════════════════════════════
#  11. SESSION STATE INITIALISATION
#  Streamlit reruns the entire script on every interaction.
#  st.session_state persists data between reruns.
# ══════════════════════════════════════════════════════════════


def init_session_state():
    defaults = {
        "stage": "start",  # start → chat → generate → results
        "current_q": 0,  # which question we are on (0-9)
        "answers": {},  # collected answers dict
        "chat_history": [],  # list of {role, content} for display
        "resume_data": None,  # generated resume dict
        "score_data": None,  # scoring results dict
        "classification": None,  # BiLSTM results
        "pdf_bytes": None,  # generated PDF bytes
        "chain": None,  # LangChain ConversationChain
        "existing_resume": "",  # optional uploaded resume text
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


# ══════════════════════════════════════════════════════════════
#  12. STREAMLIT UI
# ══════════════════════════════════════════════════════════════

st.set_page_config(page_title="AI Resume Builder", page_icon="📝", layout="wide")

init_session_state()

# ── Load classifier ───────────────────────────────────────────
clf_model, tokenizer, le, MAX_LEN = load_classifier()

# ── Setup Gemini ──────────────────────────────────────────────
gemini_model = setup_gemini()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("📝 AI Resume Builder")
    st.markdown("""
    st.write("### How It Works:")
    st.write("1. \U0001F4AC Chat with the AI assistant")
    st.write("2. \U0001F4CE Upload your existing assets or profile data")
    st.write("3. \U0001F916 Gemini writes your resume")
    st.write("4. \U0001F4C4 Download your professional ReportLab PDF")

    ---
    **Tech Stack:**
    - [Chat] LangChain conversation memory
    - Google Gemini 2.5 Flash
    - BiLSTM + Custom Attention
    - ReportLab PDF generation
    - Streamlit Cloud
    """)

    st.divider()

    # Reset button
    if st.button("🔄 Start Over", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    # Progress indicator
    if st.session_state.stage == "chat":
        q_num = st.session_state.current_q
        total = len(QUESTIONS)
        pct = int((q_num / total) * 100)
        st.markdown(f"**Progress: {q_num}/{total} questions**")
        st.progress(pct)


# ══════════════════════════════════════════════════════════════
#  STAGE 1 — START SCREEN
# ══════════════════════════════════════════════════════════════

if st.session_state.stage == "start":

    st.title("📝 AI-Powered Resume Builder")
    st.markdown(
        "Answer **10 guided questions** in a friendly chat. "
        "Gemini AI writes your professional resume. "
        "BiLSTM verifies your job category. "
        "Download as a polished PDF."
    )
    st.divider()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📎 Optional: Upload existing resume")
        st.caption("We will use it as reference context for better results.")
        uploaded = st.file_uploader(
            "Upload PDF (optional)", type=["pdf"], key="pdf_upload"
        )
        if uploaded:
            with pdfplumber.open(uploaded) as pdf:
                existing_text = "\n".join(p.extract_text() or "" for p in pdf.pages)
            st.session_state.existing_resume = existing_text
            word_count = len(existing_text.split())
            st.success(f"✅ Extracted {word_count} words from your resume")

    with col2:
        st.subheader("🚀 Start building")
        st.markdown("""
        The assistant will ask you about:
        - Personal details and contact info
        - Work experience and achievements
        - Education and skills
        - Projects and certifications
        - Target role and job description
        """)

        if st.button(
            "[Chat] Start Chat Interview", type="primary", use_container_width=True
        ):
            # Setup LangChain conversation chain
            chain = setup_langchain_chat()
            if chain:
                st.session_state.chain = chain
                st.session_state.stage = "chat"
                # First message from bot
                first_message = (
                    "Hello! 👋 I'm your AI resume assistant. "
                    "I'll guide you through 10 quick questions to build "
                    "your professional resume. Let's start!\n\n"
                    f"**Question 1/10:** {QUESTIONS[0]}"
                )
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": first_message}
                )
                st.rerun()


# ══════════════════════════════════════════════════════════════
#  STAGE 2 — CHAT INTERVIEW
#  LangChain ConversationChain handles memory.
#  Streamlit chat_message displays the conversation.
# ══════════════════════════════════════════════════════════════

elif st.session_state.stage == "chat":

    st.title("💬 Resume Interview")
    q_num = st.session_state.current_q
    total = len(QUESTIONS)

    # Progress bar
    st.progress(
        int((q_num / total) * 100), text=f"Question {q_num} of {total} answered"
    )
    st.divider()

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Chat input — only show if questions remain
    if q_num < total:
        user_input = st.chat_input(f"Your answer to question {q_num + 1}...")

        if user_input:
            # Add user message to history
            st.session_state.chat_history.append(
                {"role": "user", "content": user_input}
            )

            # Save this answer against its key
            current_key = QUESTION_KEYS[q_num]
            st.session_state.answers[current_key] = user_input

            # Advance question counter
            st.session_state.current_q += 1
            new_q_num = st.session_state.current_q

            # Generate bot response via LangChain
            with st.spinner("Thinking..."):
                try:
                    # Send user answer to LangChain chain
                    # LangChain memory automatically appends to history
                    if new_q_num < total:
                        # Build prompt with acknowledgement + next question
                        chain_input = (
                            f"User answered: {user_input}\n"
                            f"Acknowledge briefly and ask: "
                            f"{QUESTIONS[new_q_num]} "
                            f"(Question {new_q_num+1} of {total})"
                        )
                        bot_response = st.session_state.chain.predict(input=chain_input)
                    else:
                        # All questions answered
                        bot_response = (
                            "🎉 Excellent! I have all the information I need. "
                            "Let me now generate your professional resume. "
                            "This will take a few seconds..."
                        )

                except Exception as e:
                    if "429" in str(e):
                        bot_response = (
                            "⏳ Rate limit hit. Please wait 60 seconds and try again."
                        )
                    else:
                        bot_response = (
                            f"I have noted your answer. Moving to the next question."
                        )

            st.session_state.chat_history.append(
                {"role": "assistant", "content": bot_response}
            )

            # If all answered, move to generation stage
            if new_q_num >= total:
                st.session_state.stage = "generating"

            st.rerun()

    else:
        # All questions done — show generate button
        st.success("✅ All questions answered!")
        if st.button("🚀 Generate My Resume", type="primary", use_container_width=True):
            st.session_state.stage = "generating"
            st.rerun()


# ══════════════════════════════════════════════════════════════
#  STAGE 3 — GENERATE RESUME + CLASSIFY + SCORE + PDF
# ══════════════════════════════════════════════════════════════

elif st.session_state.stage == "generating":

    st.title("⚙️ Building Your Resume...")

    with st.status("Working on your resume...", expanded=True) as status:

        # Step A — Add existing resume context if uploaded
        if st.session_state.existing_resume:
            st.session_state.answers["existing_resume"] = (
                st.session_state.existing_resume[:1000]
            )

        # Step B — Generate resume content with Gemini
        st.write("🤖 Gemini is writing your resume content...")
        if gemini_model:
            resume_data = generate_resume_content(
                gemini_model, st.session_state.answers
            )
            st.session_state.resume_data = resume_data
            st.write("✅ Resume content generated")
        else:
            st.error("❌ Gemini not available")
            st.stop()

        # Step C — Classify with BiLSTM + Attention
        st.write("🧠 BiLSTM classifier verifying job category...")
        if clf_model and resume_data:
            resume_text_flat = " ".join(
                [
                    resume_data.get("summary", ""),
                    " ".join(resume_data.get("skills", {}).get("technical", [])),
                    " ".join(
                        [e.get("title", "") for e in resume_data.get("experience", [])]
                    ),
                ]
            )
            classification = classify_text(
                resume_text_flat, clf_model, tokenizer, le, MAX_LEN
            )
            st.session_state.classification = classification
            top_cat = classification[0]["category"]
            st.write(f"✅ Classified as: **{top_cat}**")
        else:
            top_cat = resume_data.get("target_role", "Professional")

        # Step D — Score the resume
        st.write("📊 Scoring resume on 10 criteria...")
        score_data = score_resume(gemini_model, resume_data, top_cat)
        st.session_state.score_data = score_data
        st.write("✅ Scoring complete")

        # Step E — Generate PDF
        st.write("📄 Generating PDF...")
        pdf_bytes = generate_pdf(resume_data)
        st.session_state.pdf_bytes = pdf_bytes
        st.write("✅ PDF ready")

        status.update(label="✅ Resume ready!", state="complete")

    st.session_state.stage = "results"
    st.rerun()


# ══════════════════════════════════════════════════════════════
#  STAGE 4 — RESULTS DASHBOARD
# ══════════════════════════════════════════════════════════════

elif st.session_state.stage == "results":

    resume_data = st.session_state.resume_data
    score_data = st.session_state.score_data
    classification = st.session_state.classification

    st.title(f"✅ Resume Ready — {resume_data.get('name','')}")
    st.divider()

    # ── Top metrics row ───────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    total_score = score_data.get("total", 0) if score_data else 0
    grade = score_data.get("grade", "N/A") if score_data else "N/A"
    top_cat = classification[0]["category"] if classification else "N/A"
    top_conf = classification[0]["confidence"] * 100 if classification else 0

    with col1:
        st.metric("Resume Score", f"{total_score}/100")
    with col2:
        st.metric("Grade", grade)
    with col3:
        st.metric("Predicted Category", top_cat)
    with col4:
        st.metric("BiLSTM Confidence", f"{top_conf:.1f}%")

    st.divider()

    # ── TABS ──────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📄 Resume Preview",
            "📊 Score Details",
            "🔍 ATS Keywords",
            "💼 LinkedIn Summary",
            "📥 Download PDF",
        ]
    )

    # TAB 1 — Resume Preview
    with tab1:
        st.subheader("Your Generated Resume")

        # Contact block
        contact_line = "  |  ".join(
            filter(
                None,
                [
                    resume_data.get("email", ""),
                    resume_data.get("phone", ""),
                    resume_data.get("location", ""),
                    resume_data.get("linkedin", ""),
                    resume_data.get("github", ""),
                ],
            )
        )
        st.markdown(
            f"<h2 style='text-align:center;color:#1a1a2e'>"
            f"{resume_data.get('name','')}</h2>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<p style='text-align:center;color:#e94560;font-weight:bold'>"
            f"{resume_data.get('target_role','')}</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<p style='text-align:center;color:gray;font-size:13px'>"
            f"{contact_line}</p>",
            unsafe_allow_html=True,
        )
        st.divider()

        # Summary
        if resume_data.get("summary"):
            st.markdown("#### 🎯 Professional Summary")
            st.markdown(resume_data["summary"])

        # Experience
        if resume_data.get("experience"):
            st.markdown("#### 💼 Work Experience")
            for exp in resume_data["experience"]:
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.markdown(
                        f"**{exp.get('title','')}** — " f"*{exp.get('company','')}*"
                    )
                with col_b:
                    st.markdown(
                        f"<p style='text-align:right;color:gray;font-size:12px'>"
                        f"{exp.get('duration','')}</p>",
                        unsafe_allow_html=True,
                    )
                for bullet in exp.get("bullets", []):
                    st.markdown(f"• {bullet}")
                st.markdown("")

        # Education
        if resume_data.get("education"):
            st.markdown("#### 🎓 Education")
            for edu in resume_data["education"]:
                st.markdown(
                    f"**{edu.get('degree','')}** — "
                    f"{edu.get('institution','')} ({edu.get('year','')})"
                )
                if edu.get("details"):
                    st.caption(edu["details"])

        # Skills
        skills = resume_data.get("skills", {})
        if skills:
            st.markdown("#### 🛠️ Skills")
            for label, key in [
                ("Technical", "technical"),
                ("Tools", "tools"),
                ("Soft Skills", "soft"),
            ]:
                items = skills.get(key, [])
                if items:
                    st.markdown(f"**{label}:** " + " · ".join(f"`{s}`" for s in items))

        # Projects
        if resume_data.get("projects"):
            st.markdown("#### 🚀 Projects")
            for proj in resume_data["projects"]:
                tech_str = " | ".join(proj.get("tech", []))
                st.markdown(
                    f"**{proj.get('name','')}** — " f"{proj.get('description','')}"
                )
                if tech_str:
                    st.caption(f"Tech: {tech_str}")

        # Certifications
        if resume_data.get("certifications"):
            st.markdown("#### 🏅 Certifications")
            for cert in resume_data["certifications"]:
                st.markdown(f"• {cert}")

    # TAB 2 — Score Details
    with tab2:
        st.subheader(f"Score: {total_score}/100  |  Grade: {grade}")

        if score_data and "scores" in score_data:
            scores = score_data["scores"]
            labels = {
                "professional_summary": "Professional Summary",
                "work_experience_quality": "Work Experience Quality",
                "skills_relevance": "Skills Relevance",
                "achievements_impact": "Achievements Impact",
                "education_presentation": "Education Presentation",
                "ats_friendliness": "ATS Friendliness",
                "action_verbs_usage": "Action Verbs Usage",
                "quantified_results": "Quantified Results",
                "overall_formatting": "Overall Formatting",
                "job_title_alignment": "Job Title Alignment",
            }
            for key, label in labels.items():
                score = scores.get(key, 0)
                pct = score * 10
                colour = "🟢" if score >= 8 else "🟡" if score >= 6 else "🔴"
                st.markdown(f"**{colour} {label}**: {score}/10")
                st.progress(pct)

            # Improvements
            improvements = score_data.get("top_improvements", [])
            if improvements:
                st.markdown("#### 💡 Top Improvements")
                for tip in improvements:
                    st.markdown(
                        f"<div style='background:#f0f7ff;border-left:"
                        f"4px solid #0f3460;padding:8px 12px;"
                        f"border-radius:0 8px 8px 0;margin-bottom:6px'>"
                        f"{tip}</div>",
                        unsafe_allow_html=True,
                    )

        # Classification results
        if classification:
            st.markdown("#### 🧠 BiLSTM Classification")
            st.caption("Verifying your resume matches your target role")
            for r in classification:
                conf = r["confidence"] * 100
                st.progress(int(conf), text=f"{r['category']} — {conf:.1f}%")

    # TAB 3 — ATS Keywords
    with tab3:
        st.subheader("ATS Keyword Analysis")
        st.markdown(
            "These keywords are critical for passing Applicant Tracking "
            "Systems (ATS). Include as many present keywords as possible."
        )

        if score_data:
            col_present, col_missing = st.columns(2)

            with col_present:
                st.markdown("#### ✅ Keywords Present")
                for kw in score_data.get("ats_keywords", []):
                    st.markdown(
                        f"<span style='background:#d4edda;color:#155724;"
                        f"padding:3px 10px;border-radius:12px;"
                        f"margin:3px;display:inline-block;font-size:13px'>"
                        f"{kw}</span>",
                        unsafe_allow_html=True,
                    )

            with col_missing:
                st.markdown("#### ❌ Keywords to Add")
                for kw in score_data.get("missing_keywords", []):
                    st.markdown(
                        f"<span style='background:#f8d7da;color:#721c24;"
                        f"padding:3px 10px;border-radius:12px;"
                        f"margin:3px;display:inline-block;font-size:13px'>"
                        f"{kw}</span>",
                        unsafe_allow_html=True,
                    )

    # TAB 4 — LinkedIn Summary
    with tab4:
        st.subheader("LinkedIn About Section")
        st.caption(
            "Copy this AI-generated LinkedIn summary to your profile. "
            "Edit it to add your personal voice."
        )
        linkedin_summary = resume_data.get("linkedin_summary", "")
        if linkedin_summary:
            st.text_area(
                "LinkedIn Summary",
                value=linkedin_summary,
                height=200,
                key="linkedin_display",
            )
            st.download_button(
                label="📋 Download LinkedIn Summary",
                data=linkedin_summary,
                file_name="linkedin_summary.txt",
                mime="text/plain",
            )
        else:
            st.info("LinkedIn summary not generated. Please try regenerating.")

    # TAB 5 — Download PDF
    with tab5:
        st.subheader("Download Your Resume")
        st.markdown(
            "Your resume has been formatted as a professional PDF. "
            "It is ATS-friendly, single-column, and ready to submit."
        )

        name_clean = resume_data.get("name", "resume").replace(" ", "_")

        if st.session_state.pdf_bytes:
            st.download_button(
                label="📥 Download Resume PDF",
                data=st.session_state.pdf_bytes,
                file_name=f"{name_clean}_resume.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="primary",
            )
            st.success(
                "✅ Your resume is ready! "
                "Use the Start Over button in the sidebar to build another."
            )
        else:
            st.error("PDF generation failed. Please start over.")

    # Footer
    st.divider()
    st.caption(
        "[Chat] LangChain conversation memory  |  "
        "Google Gemini 2.5 Flash  |  "
        "BiLSTM + Custom Attention  |  "
        "ReportLab PDF  |  "
        "Streamlit Cloud"
    )
