# YouTubeLM
Project for Media Track in DLW2025

YouTubeLM is an intelligent platform designed for YouTubers and content creators. It helps you extract insights from video data, improve content quality, and spark creative ideas. The platform retrieves video transcripts and comments from YouTube, analyzes sentiment and toxicity in comments, and performs bias analysis and fact-checking on transcripts. These analyses generate an overall integrity score and support comment classification into good and bad categories. Based on the good comments and areas of improvement (derived from bad comments and integrity scores), YouTubeLM produces a creative ideation report for your next video. Additionally, this report is refined through a regulation-aware RAG (Retrieval-Augmented Generation) module. Users can continue an interactive conversation with YouTubeLM to discuss, refine, and explore further ideas.

Pipeline
Data Retrieval

Input: A YouTube video URL provided by the user.
Processing: The system fetches the video's transcript and comments.
Data Analysis

Comments:
Perform sentiment analysis to classify comments as good or bad.
Conduct toxicity analysis on the comments.
Transcript:
Perform bias analysis.
Execute fact-checking.
Integrity Score:
Combine toxicity, bias, and fact-check results to generate an integrity score that assesses content risk.
Result Generation

Summary & Improvement:
Summarize good comments.
Use bad comments and the integrity score to generate areas for improvement.
Ideation for Next Video:
Combine the good summary and improvement areas to generate innovative video ideas for the next content piece.
The ideation report is then passed through a regulation-aware RAG module that checks against YouTube guidelines.
Interactive Dialogue

After the initial report is generated, users can interact with YouTubeLM to discuss, refine, and further optimize the report.
Main Features
Idea Generation & Improvement Suggestions: Provides innovative video concepts and detailed content improvement recommendations.
Content Regulation: Integrates bias analysis, toxicity detection, and fact-checking to produce an integrity score that helps you stay compliant with YouTube policies.
Interactive Dialogue: Supports continuous conversation, allowing users to discuss and optimize the generated report.
Multi-Source Integration: Combines video transcripts, comments, and YouTube regulations (via a RAG module) to generate comprehensive insights.
Target Audience
YouTubers and content creators looking to enhance video quality, spark creative ideas, and avoid regulatory pitfalls.
Technology Stack
Programming Language: Python
NLP & Analysis: BERT, SentenceTransformer
Fact-Checking: Fact Check API
Vector Search: FAISS
Generation Models: OpenAI API
Web Interface: Streamlit
Development Environment: Google Colab (used solely for development; end users do not need to run Colab)

## 🚀 Installation

### 1️⃣ Clone the Repository (no API key Included)
```bash
git clone <repository_url>
cd <repository_directory>


