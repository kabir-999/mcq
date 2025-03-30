import os
import fitz  # PyMuPDF for extracting text from PDFs
import json
from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
from dotenv import load_dotenv
from flask_cors import CORS

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

generation_config = {
    "temperature": 0.8,
    "top_p": 0.9,
    "top_k": 40,
    "max_output_tokens": 8192,
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
    system_instruction="Extract text from the uploaded PDF and generate 10 multiple-choice questions. "
    "Ensure the difficulty varies between easy, medium, and hard. "
    "Format:\n**Question:** <Question_Text>\n**Options:**\n(a) <Option_1>\n(b) <Option_2>\n(c) <Option_3>\n(d) <Option_4>\n**Answer:** <Correct_ans>"
)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    return text.strip()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/generate_mcqs", methods=["POST"])
def generate_mcqs():
    if "pdf_file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    pdf_file = request.files["pdf_file"]
    if pdf_file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        # Save PDF temporarily
        pdf_path = "uploaded.pdf"
        pdf_file.save(pdf_path)

        # Extract text
        extracted_text = extract_text_from_pdf(pdf_path)
        if not extracted_text:
            return jsonify({"error": "Could not extract text from the PDF"}), 400

        # Generate MCQs using Gemini AI
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(extracted_text)

        print("🔹 Raw AI Response:\n", response.text)  # Debugging

        mcqs = response.text.strip().split("\n\n")
        formatted_mcqs = []

        for mcq in mcqs:
            if "**Question:**" in mcq and "**Options:**" in mcq and "**Answer:**" in mcq:
                parts = mcq.split("**Options:**")
                question_part = parts[0].replace("**Question:**", "").strip()

                options_and_answer = parts[1].split("**Answer:**")
                options_part = options_and_answer[0].strip()
                answer_part = options_and_answer[1].strip() if len(options_and_answer) > 1 else "Answer not found"

                # Clean options
                options = [opt.strip()[4:].strip() for opt in options_part.split("\n") if opt.strip()]
                formatted_mcqs.append({"question": question_part, "options": options, "answer": answer_part})

        return jsonify({"mcqs": formatted_mcqs})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
