# CodeAlpha - Chatbot for FAQs
**Domain:** Artificial Intelligence Internship  
**Task 2:** Chatbot for FAQs  
**Repository Name:** `CodeAlpha_ChatbotForFAQs`  

---

## 📌 Project Overview
The **CodeAlpha FAQ Chatbot** is an intelligent Natural Language Processing (NLP) system designed to answer domain-specific questions with precision. It features a complete NLP preprocessing pipeline (tokenization, stopword removal, lemmatization), TF-IDF vectorization, Cosine Similarity matching, and hybrid semantic embedding ranking. It comes with an interactive Streamlit Chat UI, dynamic knowledge base manager, and a terminal CLI.

---

## ✨ Key Features
- 🧠 **Dual NLP Matching Engine:** Combines TF-IDF + Cosine Similarity with Dense Semantic Transformer Embeddings for high-accuracy semantic understanding.
- 🎯 **Confidence Scoring & Intelligent Fallbacks:** Categorizes matches into High, Moderate, and Low confidence. If a query is unclear, it proactively recommends top relevant FAQ suggestions.
- 💬 **Interactive Chat UI:** Built with Streamlit chat elements, real-time feedback badges, and one-click quick prompt chips.
- 📚 **Knowledge Base Directory:** Filter, search, and view all FAQs organized by topic categories.
- ➕ **On-the-fly Knowledge Base Expansion:** Add new question-and-answer pairs via UI forms and instantly re-index the vector spaces without restarting.
- 💻 **Terminal CLI:** Standalone interactive command-line interface for terminal usage.

---

## 📐 Mathematical Formulation: Cosine Similarity
Given a query vector $\mathbf{q}$ and a document/FAQ vector $\mathbf{d}$ in multi-dimensional TF-IDF space:

$$\text{Cosine Similarity}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\| \|\mathbf{d}\|} = \frac{\sum_{i=1}^{n} q_i d_i}{\sqrt{\sum_{i=1}^{n} q_i^2} \sqrt{\sum_{i=1}^{n} d_i^2}}$$

A similarity of `1.0` indicates identical semantic alignment, while `0.0` represents orthogonal / unrelated terms.

---

## 🛠️ Architecture & Tech Stack
- **Python 3.10+ / 3.14**
- **NLTK** (Tokenization, Stopword removal, Lemmatization)
- **Scikit-Learn** (TF-IDF Vectorizer, Cosine Similarity metric)
- **Sentence-Transformers** (Optional dense embedding matching)
- **Streamlit** (Web chat interface)

```
[ User Query ]
       │
       ▼
[ Text Preprocessing: Lowercase, Regex Clean, Stopwords Filter, Lemmatize ]
       │
       ├─────────────────────────────────┐
       ▼                                 ▼
[ TF-IDF Vectorizer ]           [ Dense Sentence Embeddings ]
       │                                 │
       ▼                                 ▼
[ Cosine Similarity Matrix ]    [ Semantic Dot Product ]
       │                                 │
       └────────────────┬────────────────┘
                        ▼
            [ Hybrid Score Fusion ]
                        │
            [ Thresholding & Fallbacks ]
            ├── High / Moderate: Return Answer + Confidence Badge
            └── Low: Suggest Top 3 Related FAQs
```

---

## 🚀 Quick Start & Installation

### 1. Clone or Navigate to the Folder
```bash
cd CodeAlpha_ChatbotForFAQs
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Web Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

### 4. Run CLI Interactive Chat
```bash
python cli.py
```

### 5. Run Unit Tests
```bash
python -m unittest tests/test_chatbot.py
```

---

## 📂 Project Structure
```
CodeAlpha_ChatbotForFAQs/
│
├── app.py                     # Streamlit Chatbot UI with search & admin tools
├── nlp_engine.py              # Text preprocessing, TF-IDF, Cosine Similarity, Embeddings
├── faqs.json                  # Curated FAQ database (CodeAlpha, AI, ML, NLP, Tech)
├── cli.py                     # Terminal conversational interface
├── requirements.txt           # Project dependencies
├── .gitignore                 # Version control exclusions
├── tests/
│   └── test_chatbot.py        # Automated test suite
└── README.md                  # Detailed documentation
```

---

## 👨‍💻 Author & Submission
- **Internship:** CodeAlpha Artificial Intelligence Internship
- **Task:** Task 2 - Chatbot for FAQs
- **Submission:** [CodeAlpha Submission Form]
