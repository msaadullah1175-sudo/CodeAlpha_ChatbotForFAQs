"""
CodeAlpha AI Internship - Task 2: Chatbot for FAQs
A simple, beginner-friendly NLP FAQ Chatbot using TF-IDF and Cosine Similarity.
"""

import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import scrolledtext

# 1. Frequently Asked Questions (FAQ) Dataset
FAQS = [
    {
        "question": "What is the CodeAlpha AI internship?",
        "answer": "The CodeAlpha AI internship provides real-world experience in AI model development, machine learning workflows, and data processing.",
    },
    {
        "question": "How many tasks do I need to complete for the certificate?",
        "answer": "To be eligible for the certificate, you must complete a minimum of 2 or 3 tasks from the task list.",
    },
    {
        "question": "How should I name my GitHub repository?",
        "answer": "You should name your repository using the format: CodeAlpha_ProjectName (e.g., CodeAlpha_ChatbotForFAQs).",
    },
    {
        "question": "Where do I submit my completed internship tasks?",
        "answer": "Submit your completed tasks using the official Google submission form provided in your WhatsApp group.",
    },
    {
        "question": "Do I need to post a video explanation on LinkedIn?",
        "answer": "Yes, post a short video explanation on LinkedIn tagging @CodeAlpha and include your GitHub repository link.",
    },
    {
        "question": "What perks will I receive after completion?",
        "answer": "You will receive an Offer Letter, QR-verified Completion Certificate, Unique ID Certificate, and a Letter of Recommendation based on performance.",
    },
    {
        "question": "What is Cosine Similarity in NLP?",
        "answer": "Cosine similarity measures the angle between two text vectors to determine how similar two questions are, ranging from 0 to 1.",
    },
    {
        "question": "What is TF-IDF?",
        "answer": "TF-IDF (Term Frequency - Inverse Document Frequency) reflects how important a word is to a document in a collection of texts.",
    },
    {
        "question": "What is an LSTM neural network?",
        "answer": "An LSTM (Long Short-Term Memory) is a type of Recurrent Neural Network designed to learn patterns from sequential data like text or music.",
    },
    {
        "question": "What is YOLO in object detection?",
        "answer": "YOLO (You Only Look Once) is an algorithm that detects objects in images and videos in real time using a single neural network pass.",
    },
]


def clean_text(text: str) -> str:
    """Simple text cleaning: lowercase and remove punctuation."""
    text = text.lower()
    return text.translate(str.maketrans("", "", string.punctuation))


# 2. Build TF-IDF Vectorizer on FAQ Questions
questions = [clean_text(faq["question"]) for faq in FAQS]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(questions)


def get_chatbot_response(user_query: str) -> str:
    """Matches user query with the most similar FAQ using Cosine Similarity."""
    cleaned_query = clean_text(user_query)
    if not cleaned_query.strip():
        return "Please ask a question!"

    # Transform query and compute cosine similarity against all FAQ questions
    query_vector = vectorizer.transform([cleaned_query])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    best_match_idx = similarity_scores.argmax()
    best_score = similarity_scores[best_match_idx]

    # Confidence threshold: 0.25
    if best_score >= 0.25:
        return FAQS[best_match_idx]["answer"]
    else:
        return (
            "I'm sorry, I don't know the answer to that question.\n"
            "You can ask me about: internship requirements, tasks, certificates, GitHub repo naming, or AI concepts!"
        )


# 3. Simple Desktop GUI using Tkinter
def send_message():
    user_text = entry_box.get().strip()
    if not user_text:
        return

    chat_area.config(state=tk.NORMAL)
    chat_area.insert(tk.END, f"You: {user_text}\n", "user")

    response = get_chatbot_response(user_text)
    chat_area.insert(tk.END, f"Bot: {response}\n\n", "bot")
    chat_area.config(state=tk.DISABLED)
    chat_area.yview(tk.END)
    entry_box.delete(0, tk.END)


def run_gui():
    root = tk.Tk()
    root.title("CodeAlpha - FAQ Chatbot")
    root.geometry("540x520")
    root.configure(bg="#f8fafc")

    tk.Label(
        root,
        text="🤖 CodeAlpha FAQ Chatbot",
        font=("Arial", 14, "bold"),
        bg="#f8fafc",
        fg="#0f172a",
    ).pack(pady=10)

    global chat_area, entry_box
    chat_area = scrolledtext.ScrolledText(
        root, wrap=tk.WORD, width=58, height=20, font=("Arial", 10)
    )
    chat_area.pack(padx=15, pady=5)
    chat_area.tag_config("user", foreground="#2563eb", font=("Arial", 10, "bold"))
    chat_area.tag_config("bot", foreground="#16a34a")

    chat_area.insert(
        tk.END,
        "Bot: Hello! Ask me any question about the CodeAlpha internship or AI topics.\n\n",
        "bot",
    )
    chat_area.config(state=tk.DISABLED)

    bottom_frame = tk.Frame(root, bg="#f8fafc")
    bottom_frame.pack(padx=15, pady=10, fill=tk.X)

    entry_box = tk.Entry(bottom_frame, font=("Arial", 11), width=40)
    entry_box.pack(side=tk.LEFT, padx=(0, 10), ipady=3)
    entry_box.bind("<Return>", lambda event: send_message())

    send_btn = tk.Button(
        bottom_frame,
        text="Send",
        command=send_message,
        bg="#2563eb",
        fg="white",
        font=("Arial", 10, "bold"),
        padx=15,
        pady=3,
        cursor="hand2",
    )
    send_btn.pack(side=tk.LEFT)

    root.mainloop()


if __name__ == "__main__":
    # If run in terminal, you can also test terminal chat or GUI
    import sys

    if "--cli" in sys.argv:
        print("=== CodeAlpha FAQ Chatbot (Terminal Mode) ===")
        print("Type 'exit' to quit.\n")
        while True:
            q = input("You: ").strip()
            if q.lower() in ["exit", "quit"]:
                break
            print(f"Bot: {get_chatbot_response(q)}\n")
    else:
        run_gui()
