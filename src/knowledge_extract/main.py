import os
import markdown
from dotenv import load_dotenv
from src.knowledge_extract.crew import KnowledgeExtract

def run_loop():
    """
    Runs the main question-and-answer loop, collects all results,
    and generates a single HTML summary at the end.
    """
    print("\n📚 CrewAI Doc QA — ask questions about your PDFs.")
    print("Type 'exit' to quit.\n")

    # Initialize the CrewAI application
    crew_app = KnowledgeExtract()
    crew = crew_app.crew()

    # List to store the markdown table for each Q&A pair
    qa_history = []

    while True:
        try:
            question = input("❓ Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            # Handle Ctrl+C or Ctrl+D to exit gracefully
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", ":q"}:
            break

        try:
            print("\n🚀 Kicking off crew to answer your question...")
            # Attempt to get an answer from the CrewAI crew
            crew_output = crew.kickoff(inputs={"question": question})
            print("\n--- Answer ---")
            print(crew_output)
            print("--------------\n")

            # --- Data Cleaning ---
            raw_output = str(crew_output).strip()
            cleaned_markdown = raw_output
            # Remove markdown code fences if they exist
            if raw_output.startswith("```") and raw_output.endswith("```"):
                lines = raw_output.split('\n')
                cleaned_markdown = '\n'.join(lines[1:-1])

            # Add the successful result to our history
            qa_history.append(cleaned_markdown)
            print("✅ Answer recorded. Ask another question or type 'exit'.")

        except Exception as e:
            # If any error occurs (like a server overload), catch it
            print(f"\n❌ An error occurred: {e}")
            # Create a user-friendly error table and add it to the history
            error_table = (
                f"| Question | Answer |\n"
                f"|---|---|\n"
                f"| {question} | **Error:** The AI model was overloaded or an error occurred. Please try again later. |"
            )
            qa_history.append(error_table)

    # --- This section runs only after the user types 'exit' ---
    print("\n👋 Bye! Generating your final Q&A report...")

    if not qa_history:
        print("No questions were answered. Exiting.")
        return

    # Join all the collected markdown tables, separated by a horizontal rule
    full_markdown_content = "\n<hr>\n".join(qa_history)
    html_body_content = markdown.markdown(full_markdown_content, extensions=['tables'])

    # --- HTML Page Generation ---
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>CrewAI Q&A Summary</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                margin: 40px;
                background-color: #f9f9f9;
                color: #333;
            }}
            h1 {{
                text-align: center;
                color: #2c3e50;
            }}
            table {{
                border-collapse: collapse;
                width: 90%;
                max-width: 800px;
                margin: 25px auto;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                border-radius: 8px;
                overflow: hidden;
            }}
            th, td {{
                border: 1px solid #ddd;
                text-align: left;
                padding: 14px;
            }}
            th {{
                background-color: #4CAF50;
                color: white;
            }}
            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            hr {{
                border: 0;
                height: 1px;
                background: #ccc;
                margin: 40px auto;
                width: 90%;
            }}
        </style>
    </head>
    <body>
        <h1>Q&A Summary</h1>
        {html_body_content}
    </body>
    </html>
    """

    output_filename = "output.html"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(html_template)

    print(f"✅ Successfully saved the full conversation to {output_filename}")

if __name__ == "__main__":
    load_dotenv()
    # Ensure the 'knowledge' directory exists
    kb_dir = os.path.join(os.getcwd(), "knowledge")
    if not os.path.isdir(kb_dir):
        print(f"⚠  Missing folder: {kb_dir}. Create it and add your PDFs.")
    run_loop()
