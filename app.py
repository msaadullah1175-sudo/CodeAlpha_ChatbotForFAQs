"""
CodeAlpha Artificial Intelligence Internship
TASK 2: Chatbot for FAQs
Interactive Web Application - Hugging Face AI answers EVERYTHING
"""

import os
import streamlit as st
from nlp_engine import FAQMatcher

st.set_page_config(
    page_title="CodeAlpha | AI FAQ Assistant",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ──────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .chat-header {
        background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 50%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 0.2rem;
    }

    .pill-high {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 9999px;
        padding: 0.15rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .pill-mod {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 9999px;
        padding: 0.15rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .pill-low {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 9999px;
        padding: 0.15rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .pill-hf {
        background: rgba(139, 92, 246, 0.15);
        color: #a78bfa;
        border: 1px solid rgba(139, 92, 246, 0.3);
        border-radius: 9999px;
        padding: 0.15rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Cache Resources ───────────────────────────────────────────────────────────
@st.cache_resource
def get_faq_matcher():
    return FAQMatcher()

matcher = get_faq_matcher()

# ── Chat History Init ─────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 Hello! I'm the **CodeAlpha AI Assistant** powered by **Hugging Face LLMs**.\n\n"
                "I can answer **anything** — coding, science, AI, math, or CodeAlpha internship questions!\n\n"
                "🔑 Add your free Hugging Face token in the sidebar to unlock full AI power."
            ),
            "confidence": 1.0,
            "confidence_label": "High",
            "source": "Greeting",
            "suggestions": [],
        }
    ]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🤖 CodeAlpha AI Internship")
    st.markdown("#### Task 2: FAQ Chatbot with Hugging Face AI")
    st.write("Powered by **Hugging Face LLM APIs** — answers any question on any topic!")
    st.markdown("---")

    from hf_client import SUPPORTED_HF_MODELS

    st.subheader("🤗 Hugging Face AI Setup")

    hf_token_val = st.text_input(
        "Hugging Face API Token",
        type="password",
        value=os.getenv("HF_TOKEN", ""),
        placeholder="hf_xxxxxxxxxxxxxxxxxxxx",
        help="Get a free token at https://huggingface.co/settings/tokens",
    )

    model_label = st.selectbox("AI Model", list(SUPPORTED_HF_MODELS.keys()), index=0)
    chosen_model = SUPPORTED_HF_MODELS[model_label]

    if hf_token_val.strip():
        st.success(f"🟢 Connected · {model_label}")
    else:
        st.warning(
            "⚠️ No token — AI will only answer FAQ questions.  \n"
            "[Get a free token here](https://huggingface.co/settings/tokens)"
        )

    # Always use HF for everything toggle
    always_hf = st.checkbox(
        "Always use Hugging Face AI (even for FAQ matches)",
        value=True,
        help="When ON, every question goes to Hugging Face for a rich AI answer.",
    )

    st.markdown("---")
    st.metric("Knowledge Base FAQs", len(matcher.faqs))

    st.markdown("---")
    st.subheader("💡 Try asking...")
    quick_prompts = [
        "Explain neural networks simply",
        "Write Python code for bubble sort",
        "What is the capital of Japan?",
        "How does YOLO object detection work?",
        "What is quantum entanglement?",
        "How many tasks in the CodeAlpha internship?",
    ]
    for qp in quick_prompts:
        if st.button(qp, key=f"qp_{qp}"):
            st.session_state.user_query_input = qp

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Chat cleared! Ask me anything.",
                "confidence": 1.0,
                "confidence_label": "High",
                "source": "System",
                "suggestions": [],
            }
        ]
        st.rerun()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="chat-header">💬 CodeAlpha AI Chatbot</div>', unsafe_allow_html=True)
st.caption("Powered by Hugging Face LLMs — ask anything from coding to science to CodeAlpha internship!")

tab_chat, tab_kb, tab_add = st.tabs(
    ["💬 Chat", "📚 Knowledge Base", "➕ Add FAQ"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("source"):
                src = msg.get("source", "")
                label = msg.get("confidence_label", "")
                if "Hugging" in src or "Hugging" in label:
                    pill = "pill-hf"
                elif "High" in label:
                    pill = "pill-high"
                elif "Moderate" in label:
                    pill = "pill-mod"
                else:
                    pill = "pill-low"
                st.markdown(f'<span class="{pill}">{src}</span>', unsafe_allow_html=True)
            if msg.get("suggestions"):
                st.markdown("**Related Questions:**")
                for s in msg["suggestions"]:
                    st.markdown(f"- *{s}*")

    # Get user input
    default_text = st.session_state.pop("user_query_input", None)
    prompt = st.chat_input("Ask me anything — coding, science, AI, CodeAlpha...") or default_text

    if prompt:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_data = matcher.match(
                    prompt,
                    hf_token=hf_token_val.strip(),
                    model_name=chosen_model,
                    always_use_hf=always_hf,
                )

            st.markdown(response_data["answer"])

            src = response_data.get("source", "")
            label = response_data.get("confidence_label", "")
            if "Hugging" in src or "Hugging" in label:
                pill = "pill-hf"
            elif "High" in label:
                pill = "pill-high"
            elif "Moderate" in label:
                pill = "pill-mod"
            else:
                pill = "pill-low"
            st.markdown(f'<span class="{pill}">{src}</span>', unsafe_allow_html=True)

            if response_data.get("suggestions"):
                st.markdown("**Related Questions:**")
                for s in response_data["suggestions"]:
                    st.markdown(f"- *{s}*")

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response_data["answer"],
                "confidence": response_data["confidence"],
                "confidence_label": response_data["confidence_label"],
                "source": response_data.get("source", ""),
                "suggestions": response_data["suggestions"],
            }
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — KNOWLEDGE BASE
# ══════════════════════════════════════════════════════════════════════════════
with tab_kb:
    st.subheader("📚 Browse Knowledge Base")
    search_term = st.text_input("🔍 Search FAQs", placeholder="e.g. certificate, yolo, tasks")
    categories = sorted({faq.get("category", "General") for faq in matcher.faqs})
    selected_cat = st.selectbox("Filter by Category", ["All Categories"] + categories)

    filtered = matcher.faqs
    if selected_cat != "All Categories":
        filtered = [f for f in filtered if f.get("category") == selected_cat]
    if search_term.strip():
        kw = search_term.lower()
        filtered = [
            f for f in filtered
            if kw in f["question"].lower() or kw in f["answer"].lower()
            or any(kw in t.lower() for t in f.get("tags", []))
        ]

    st.write(f"Showing **{len(filtered)}** FAQs")
    for faq in filtered:
        with st.expander(f"📌 {faq['question']} ({faq.get('category', 'General')})"):
            st.markdown(f"**Answer:** {faq['answer']}")
            tags = " ".join([f"`#{t}`" for t in faq.get("tags", [])])
            st.caption(f"Tags: {tags}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ADD FAQ
# ══════════════════════════════════════════════════════════════════════════════
with tab_add:
    st.subheader("➕ Add New FAQ to Knowledge Base")
    with st.form("add_faq_form"):
        new_cat = st.selectbox(
            "Category",
            ["CodeAlpha Internship", "Artificial Intelligence", "Technical Support", "Other"],
        )
        new_q = st.text_input("Question", placeholder="e.g. What is the LOR criteria?")
        new_a = st.text_area("Answer", placeholder="e.g. LOR is awarded based on...")
        new_tags = st.text_input("Tags (comma-separated)", placeholder="lor, criteria, recommendation")

        if st.form_submit_button("💾 Save FAQ"):
            if not new_q.strip() or not new_a.strip():
                st.error("Please fill in both Question and Answer.")
            else:
                tag_list = [t.strip().lower() for t in new_tags.split(",") if t.strip()]
                if matcher.add_faq(new_q, new_a, new_cat, tag_list):
                    st.success("✅ FAQ added and knowledge base updated!")
                else:
                    st.error("Failed to save FAQ.")
