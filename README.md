# YouTubeLM
Project for Media Track in DLW2025

The **YouTubeLM** is a **Streamlit** application designed to analyze YouTube videos and provide insights on content integrity and review sentiment. The app retrieves video data (including transcript and comments), computes sub-scores for **bias, misinformation, and toxicity**, and calculates an **overall integrity score** with an explanation. It also includes interactive visualizations and a conversational AI chatbot that delivers content strategy proposals and answers follow-up questions.

---

## 📌 Features

### 🎥 **Video Analysis**
- Retrieves **video metadata** from a YouTube URL.
- Computes sub-scores: **Bias**, **Misinformation**, and **Toxicity**.
- Calculates an **Integrity Score** and provides a textual explanation.
- Performs **sentiment analysis** on comments, categorizing them as **good** or **bad**.

### 💬 **Interactive Chatbot**
- Provides an **AI-powered chat interface** with OpenAI’s GPT model.
- Generates a **detailed proposal** based on video analysis.
- Allows **continuous chat interaction**, with:
  - A **scrollable chat history**.
  - A **sticky input box** at the bottom.
  - **Hides system prompts** from chat history.

---

## 🚀 Installation

### 1️⃣ Clone the Repository (no API key Included)
```bash
git clone <repository_url>
cd <repository_directory>


