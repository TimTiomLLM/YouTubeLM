# YouTubeLM
Project for Media Track in DLW2025

## For Judges
This hackathon project is structured to allow quick evaluation. All core functions and helper modules are located in `youtubelm_functions.py`, while the main execution logic and Streamlit web interface are implemented in `app.py`. Please refer to these files for a detailed understanding of the project's functionality.

## Introduction
YouTubeLM is an intelligent platform designed for YouTubers and content creators. It extracts insights from video data to improve content quality and spark creative ideas. The platform retrieves video transcripts and comments from YouTube, analyzes sentiment and toxicity in comments, and performs bias analysis and fact-checking on transcripts. These analyses generate an overall integrity score and help classify comments into good and bad categories. Based on good comments and identified areas of improvement (derived from bad comments and integrity scores), YouTubeLM produces a creative ideation report for your next video. Additionally, this report is refined through a regulation-aware RAG (Retrieval-Augmented Generation) module. Users can also engage in an interactive conversation with YouTubeLM to discuss, refine, and explore further ideas.

---

## Pipeline

1. **Data Retrieval**  
   - **Input:** A YouTube video URL provided by the user.  
   - **Processing:** The system fetches the video transcript and comments.

2. **Data Analysis**  
   - **Comments:**  
     - Sentiment analysis to classify comments as good or bad.  
     - Toxicity analysis on comments.
   - **Transcript:**  
     - Bias analysis.  
     - Fact-checking.
   - **Integrity Score:**  
     - Combines toxicity, bias, and fact-check results to generate an integrity score that assesses content risk.

3. **Result Generation**  
   - **Summary & Improvement:**  
     - Good comments are summarized.  
     - Bad comments and the integrity score are used to generate areas for improvement.
   - **Ideation for Next Video:**  
     - Combines the good summary and improvement areas to generate innovative ideas for the next video.  
     - The ideation report is further refined by a regulation-aware RAG module that checks against YouTube guidelines.

4. **Interactive Dialogue**  
   - After generating the initial report, users can continue interacting with YouTubeLM via a web interface to discuss, refine, and further optimize the report.

---

## Main Features

- **Idea Generation & Improvement Suggestions:** Provides innovative video concepts and detailed content improvement recommendations.
- **Content Regulation:** Integrates bias analysis, toxicity detection, and fact-checking to produce an integrity score that ensures compliance with YouTube policies.
- **Interactive Dialogue:** Supports continuous conversation, allowing users to discuss and optimize the generated report.
- **Multi-Source Integration:** Combines video transcripts, comments, and YouTube regulations (via a RAG module) to generate comprehensive insights.

---

## Target Audience

- **YouTubers and Content Creators:** Enhance video quality, spark creative ideas, and avoid regulatory pitfalls.

---

## Technology Stack

- **Programming Language:** Python  
- **NLP & Analysis:** BERT, SentenceTransformer  
- **Fact-Checking:** Fact Check API  
- **Vector Search:** FAISS  
- **Generation Models:** OpenAI API  
- **Web Interface & Deployment:** Streamlit  

---

## Installation & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/YouTubeLM.git
cd YouTubeLM
```

### 2. Install Dependencies
All required dependencies are listed in requirements.txt. Install them with:

```bash
pip install -r requirements.txt
```

### 3. Run the Project
YouTubeLM is deployed as a web application using Streamlit. Start the application with:

```bash
streamlit run app.py
```
This command launches YouTubeLM in your web browser, where you can enter a YouTube video URL and interact with the platform.

### 4. Input Video URL
When prompted, enter a YouTube video URL. The system will:

- **Fetch the video transcript and comments.
- **Perform analysis and generate score visualizations.
- **Produce a creative ideation report for your next video.
- **Allow you to engage in an interactive dialogue to further discuss and refine the report


