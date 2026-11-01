"""
CodeAlpha Artificial Intelligence Internship
TASK 2: Chatbot for FAQs - Interactive Command-Line Interface
"""

import sys
from nlp_engine import FAQMatcher


def main():
    print("=" * 65)
    print("🤖 CodeAlpha AI - FAQ Chatbot (Terminal Interactive Mode)")
    print("Ask questions about CodeAlpha, AI concepts, or NLP.")
    print("Type 'exit' or 'quit' to end the session.")
    print("=" * 65)

    matcher = FAQMatcher()
    print(f"Loaded {len(matcher.faqs)} knowledge base entries.\n")

    while True:
        try:
            query = input("You > ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                print("\nGoodbye! Best of luck with your CodeAlpha internship!")
                break

            result = matcher.match(query)
            print(f"\nAssistant [{result['confidence_label']} confidence | {result['confidence']:.0%}]:")
            print(f"{result['answer']}\n")

            if result.get("suggestions"):
                print("Suggested related questions:")
                for sug in result["suggestions"]:
                    print(f"  • {sug}")
                print()

        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()
