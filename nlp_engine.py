"""
CodeAlpha Artificial Intelligence Internship
TASK 2: Chatbot for FAQs - NLP Processing & Matching Engine
"""

import json
import os
import re
import string
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Stop words and lemmatizer setup (100% offline and robust)
STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it", "its",
    "itself", "me", "more", "most", "my", "myself", "no", "nor", "not", "of",
    "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves",
    "out", "over", "own", "same", "she", "should", "shouldn't", "so", "some",
    "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then",
    "there", "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "wasn't", "we", "were", "weren't", "what",
    "when", "where", "which", "while", "who", "whom", "why", "with", "would",
    "you", "your", "yours", "yourself", "yourselves"
}

LEMMATIZER = None
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    try:
        if nltk.data.find("corpora/stopwords"):
            STOP_WORDS = set(stopwords.words("english"))
    except Exception:
        pass
    try:
        if nltk.data.find("corpora/wordnet"):
            LEMMATIZER = WordNetLemmatizer()
    except Exception:
        pass
except Exception:
    pass


def preprocess_text(text: str) -> str:
    """
    Cleans, tokenizes, removes punctuation/stopwords, and lemmatizes input text.
    """
    if not text:
        return ""

    # Convert to lowercase
    text = text.lower().strip()

    # Remove URLs, HTML tags, punctuation
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenize by whitespace and alphanumeric chars
    tokens = [tok for tok in text.split() if tok.isalnum()]

    # Filter stopwords and short tokens
    cleaned_tokens = [tok for tok in tokens if tok not in STOP_WORDS and len(tok) > 1]

    # Lemmatize if lemmatizer is available
    if LEMMATIZER:
        try:
            cleaned_tokens = [LEMMATIZER.lemmatize(tok) for tok in cleaned_tokens]
        except Exception:
            pass

    return " ".join(cleaned_tokens) if cleaned_tokens else text


class FAQMatcher:
    """
    Matches user questions against an FAQ knowledge base using:
    1. Exact / Substring Intent Checking
    2. TF-IDF Vectorizer with Cosine Similarity
    3. Semantic Dense Embeddings (via sentence-transformers if available)
    """

    def __init__(self, faq_file_path: Optional[str] = None, enable_embeddings: bool = False):
        self.faq_file = faq_file_path or os.path.join(
            os.path.dirname(__file__), "faqs.json"
        )
        self.enable_embeddings = enable_embeddings
        self.faqs: List[Dict[str, Any]] = []
        self.processed_corpus: List[str] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.embedding_model = None
        self.faq_embeddings = None

        self._load_faqs()
        self._initialize_models()

    def _load_faqs(self):
        """Loads FAQs from the JSON knowledge base."""
        if os.path.exists(self.faq_file):
            try:
                with open(self.faq_file, "r", encoding="utf-8") as f:
                    self.faqs = json.load(f)
            except Exception as e:
                print(f"Error loading FAQs from {self.faq_file}: {e}")
                self.faqs = []
        else:
            self.faqs = []

    def _initialize_models(self):
        """Preprocesses FAQ corpus, builds TF-IDF matrix."""
        if not self.faqs:
            return

        # Prepare corpus: combine question, tags, and category for rich context
        self.processed_corpus = []
        for faq in self.faqs:
            tags_str = " ".join(faq.get("tags", []))
            full_context = f"{faq['question']} {tags_str} {faq.get('category', '')}"
            self.processed_corpus.append(preprocess_text(full_context))

        # TF-IDF Vectorization & Matrix calculation (Sub-millisecond)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.tfidf_matrix = self.vectorizer.fit_transform(self.processed_corpus)

        # Optional Semantic Embedding loading (only if explicitly enabled)
        if self.enable_embeddings:
            try:
                from sentence_transformers import SentenceTransformer

                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                faq_texts = [faq["question"] for faq in self.faqs]
                self.faq_embeddings = self.embedding_model.encode(
                    faq_texts, convert_to_numpy=True, normalize_embeddings=True
                )
            except Exception:
                self.embedding_model = None
                self.faq_embeddings = None

    def match(
        self,
        query: str,
        top_k: int = 3,
        hf_token: Optional[str] = None,
        model_name: Optional[str] = None,
        always_use_hf: bool = True,
    ) -> Dict[str, Any]:
        """
        Takes a raw user query and returns an answer.
        When always_use_hf=True AND a token is provided, every question goes
        straight to Hugging Face LLM for a rich AI answer.
        Falls back to local FAQ matching when HF is not configured.
        """
        clean_query = query.strip()
        if not clean_query:
            return {
                "matched": False,
                "answer": "Please ask a question so I can assist you!",
                "confidence": 0.0,
                "confidence_label": "None",
                "source": "None",
                "matched_faq": None,
                "suggestions": [],
            }

        # Greeting shortcut
        greetings = {"hello", "hi", "hey", "greetings", "good morning", "good evening"}
        if clean_query.lower() in greetings:
            return {
                "matched": True,
                "answer": "Hello! I'm your CodeAlpha AI Assistant powered by Hugging Face LLMs. Ask me anything!",
                "confidence": 1.0,
                "confidence_label": "High",
                "source": "Greeting",
                "matched_faq": None,
                "suggestions": [faq["question"] for faq in self.faqs[:3]],
            }

        # ── Configure HF client ──────────────────────────────────────────────
        from hf_client import hf_client
        if hf_token:
            hf_client.set_token(hf_token)
        if model_name:
            hf_client.set_model(model_name)

        # ── Always-HF mode: route everything to Hugging Face first ──────────
        if always_use_hf and hf_client.is_configured():
            # Compute FAQ suggestions in background for "Related Questions"
            suggestions = []
            if self.faqs and self.vectorizer is not None:
                processed_q = preprocess_text(clean_query) or clean_query.lower()
                q_vec = self.vectorizer.transform([processed_q])
                scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
                ranked = np.argsort(scores)[::-1]
                for idx in ranked[:top_k]:
                    if float(scores[int(idx)]) > 0.05:
                        suggestions.append(self.faqs[int(idx)]["question"])

            hf_result = hf_client.generate_response(clean_query)
            if hf_result["success"]:
                return {
                    "matched": True,
                    "answer": hf_result["response"],
                    "confidence": 0.97,
                    "confidence_label": "Hugging Face AI",
                    "source": f"🤗 Hugging Face · {hf_result['model'].split('/')[-1]}",
                    "matched_faq": None,
                    "suggestions": suggestions,
                }
            else:
                # HF failed (e.g. rate limit) — fall through to local FAQ
                pass

        # ── Local FAQ matching (TF-IDF + Cosine Similarity) ─────────────────
        if not self.faqs or self.vectorizer is None:
            return {
                "matched": False,
                "answer": (
                    "No knowledge base loaded and Hugging Face token is not configured.\n\n"
                    "💡 Add your free **Hugging Face API token** in the sidebar to enable AI answers!"
                ),
                "confidence": 0.0,
                "confidence_label": "Low",
                "source": "None",
                "matched_faq": None,
                "suggestions": [],
            }

        processed_q = preprocess_text(clean_query) or clean_query.lower()
        q_vec = self.vectorizer.transform([processed_q])
        tfidf_scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        ranked_indices = np.argsort(tfidf_scores)[::-1]
        best_idx = int(ranked_indices[0])
        best_score = float(tfidf_scores[best_idx])
        best_faq = self.faqs[best_idx]

        suggestions = []
        for idx in ranked_indices[1 : top_k + 1]:
            if float(tfidf_scores[int(idx)]) > 0.05:
                suggestions.append(self.faqs[int(idx)]["question"])

        # High confidence FAQ match
        if best_score >= 0.40:
            return {
                "matched": True,
                "answer": best_faq["answer"],
                "confidence": round(best_score, 3),
                "confidence_label": "High",
                "source": "Knowledge Base (FAQ)",
                "matched_faq": best_faq,
                "suggestions": suggestions,
            }

        # Moderate match — try HF as upgrade if available
        if hf_client.is_configured():
            hf_result = hf_client.generate_response(clean_query)
            if hf_result["success"]:
                return {
                    "matched": True,
                    "answer": hf_result["response"],
                    "confidence": 0.95,
                    "confidence_label": "Hugging Face AI",
                    "source": f"🤗 Hugging Face · {hf_result['model'].split('/')[-1]}",
                    "matched_faq": None,
                    "suggestions": suggestions[:2],
                }

        if best_score >= 0.20:
            return {
                "matched": True,
                "answer": f"Here is what might help:\n\n{best_faq['answer']}",
                "confidence": round(best_score, 3),
                "confidence_label": "Moderate",
                "source": "Knowledge Base (Partial Match)",
                "matched_faq": best_faq,
                "suggestions": suggestions,
            }

        # Low confidence — no HF token
        return {
            "matched": False,
            "answer": (
                "I couldn't find a strong match in the knowledge base.\n\n"
                "💡 **Add your free Hugging Face token** in the sidebar so I can answer **anything**!"
            ),
            "confidence": round(best_score, 3),
            "confidence_label": "Low",
            "source": "None",
            "matched_faq": None,
            "suggestions": [self.faqs[int(idx)]["question"] for idx in ranked_indices[:top_k]],
        }

    def add_faq(self, question: str, answer: str, category: str = "General", tags: Optional[List[str]] = None) -> bool:
        """Adds a new FAQ entry to knowledge base and updates indices."""
        new_faq = {
            "id": len(self.faqs) + 1,
            "category": category,
            "question": question.strip(),
            "answer": answer.strip(),
            "tags": tags or [t.lower() for t in question.split()[:4]],
        }
        self.faqs.append(new_faq)
        try:
            with open(self.faq_file, "w", encoding="utf-8") as f:
                json.dump(self.faqs, f, indent=2)
            self._initialize_models()
            return True
        except Exception as e:
            print(f"Error saving updated FAQs: {e}")
            return False

