import os
import re
import csv
import openai
import torch
import nltk
import numpy as np
import faiss
import PyPDF2

from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TextClassificationPipeline
)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer

# Download required resources
nltk.download('vader_lexicon', quiet=True)

######################################
# CONSTANTS & API KEYS
######################################
YOUTUBE_API_KEY = "AIzaSyAdfy19rpxdPiWzFdEanseFQb68HH9oheg"
OPENAI_API_KEY = "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"

openai.api_key = OPENAI_API_KEY

######################################
# YOUTUBE DATA RETRIEVAL FUNCTIONS
######################################

def extract_video_id(url):
    """
    Extract the video ID from a YouTube URL.
    """
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, url)
    if match:
        return match.group(1)
    else:
        raise ValueError("Invalid YouTube URL provided.")

def get_youtube_service():
    """Return a YouTube service object using the API key."""
    return build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

def fetch_video_title(video_id):
    """Retrieve the video title via YouTube Data API."""
    youtube = get_youtube_service()
    request = youtube.videos().list(part="snippet", id=video_id)
    response = request.execute()
    if response.get("items"):
        return response["items"][0]["snippet"]["title"]
    return None

def fetch_video_comments(video_id, max_results=100):
    """Fetch up to max_results comments for a given video."""
    youtube = get_youtube_service()
    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_results,
        textFormat="plainText"
    )
    response = request.execute()
    while True:
        for item in response.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
        if "nextPageToken" in response:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=max_results,
                pageToken=response["nextPageToken"],
                textFormat="plainText"
            )
            response = request.execute()
        else:
            break
    return comments

def fetch_video_transcript(video_id):
    """
    Retrieve the video transcript using YouTubeTranscriptApi.
    Returns None if unavailable.
    """
    try:
        segments = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = " ".join(segment["text"] for segment in segments)
        return transcript
    except Exception as e:
        print(f"Transcript not available via API: {e}")
        return None

def analyze_video(video_url):
    """
    Given a YouTube URL, return a dict with video_id, title, transcript, and comments.
    """
    video_id = extract_video_id(video_url)
    title = fetch_video_title(video_id)
    transcript = fetch_video_transcript(video_id)
    comments = fetch_video_comments(video_id)
    return {"video_id": video_id, "title": title, "transcript": transcript, "comments": comments}

######################################
# CSV HELPER FUNCTIONS
######################################

def save_comments_to_csv(comments, filename="video_comments.csv"):
    """
    Save a list of comment strings to a CSV file with header 'comment'.
    """
    with open(filename, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["comment"])
        for comment in comments:
            writer.writerow([comment])
    print(f"Saved {len(comments)} comments to {filename}")

def load_comments_from_csv(filename="video_comments.csv"):
    """
    Load comments from a CSV file (assumes header 'comment').
    """
    comments = []
    with open(filename, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            comments.append(row["comment"])
    return comments

######################################
# SCORING SYSTEM FUNCTIONS
######################################

# Initialize the media bias model
_bias_tokenizer = AutoTokenizer.from_pretrained("rinapch/distilbert-media-bias")
_bias_model = AutoModelForSequenceClassification.from_pretrained("rinapch/distilbert-media-bias")

def calculate_bias_score(transcript):
    """
    Calculate bias score (neutral probability) from transcript.
    Returns 0.5 if transcript is empty.
    """
    if not transcript:
        return 0.5
    inputs = _bias_tokenizer(transcript, return_tensors="pt", truncation=True)
    outputs = _bias_model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    neutral_probability = probabilities[0][1].item()  # assume index 1 is "Center"
    return max(0, min(1, neutral_probability))

def calculate_misinformation_score(transcript):
    """
    Dummy heuristic: calculate score based on count of sentences with numerical claims.
    """
    if not transcript:
        return 0.5
    sentences = transcript.split(".")
    claim_sentences = [s for s in sentences if re.search(r"\d+", s)]
    score = len(claim_sentences) / (len(sentences) + 1)
    return min(1, score)

def calculate_toxic_score(comments):
    """
    Calculate average non-toxicity score from comments using a toxic-comment model.
    """
    model_path = "martin-ha/toxic-comment-model"
    tox_tokenizer = AutoTokenizer.from_pretrained(model_path)
    tox_model = AutoModelForSequenceClassification.from_pretrained(model_path)
    pipeline = TextClassificationPipeline(model=tox_model, tokenizer=tox_tokenizer, return_all_scores=True)
    if not comments:
        return 0.5
    total_score = 0.0
    for comment in comments:
        results = pipeline(comment)
        if results and isinstance(results[0], list):
            results = results[0]
        non_toxic_score = None
        for res in results:
            if res['label'].lower() in ['non_toxic', 'non-toxic', 'clean']:
                non_toxic_score = res['score']
                break
        if non_toxic_score is None:
            for res in results:
                if res['label'].lower() == 'toxic':
                    non_toxic_score = 1 - res['score']
                    break
        if non_toxic_score is None:
            non_toxic_score = 0.5
        total_score += non_toxic_score
    return total_score / len(comments)

def calculate_integrity_score(bias_score, misinformation_score, toxic_score):
    """
    Combine sub-scores into an overall integrity score (1-10) using weighted factors.
    """
    weighted = (bias_score * 0.4) + (misinformation_score * 0.4) + (toxic_score * 0.2)
    integrity_score = 1 + (weighted * 9)
    explanation = (f"Weighted calculation: bias={bias_score:.2f} (40%), "
                   f"misinformation={misinformation_score:.2f} (40%), "
                   f"toxicity={toxic_score:.2f} (20%). "
                   f"Final integrity score: {integrity_score:.2f}/10.")
    return integrity_score, explanation

######################################
# SENTIMENT ANALYSIS & REVIEW FUNCTIONS
######################################

# Initialize sentiment analysis model
_sentiment_tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
_sentiment_model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

def analyze_sentiment(text):
    """
    Get sentiment score (1-5) for text using a BERT-based sentiment model.
    """
    inputs = _sentiment_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = _sentiment_model(**inputs)
    scores = outputs.logits.softmax(dim=1)
    return torch.argmax(scores).item() + 1

def categorize_and_save_reviews(input_csv, good_csv, bad_csv):
    """
    Categorize comments from input CSV into good (score ≥ 3) and bad reviews.
    Saves results into separate CSV files.
    """
    with open(input_csv, "r", encoding="utf-8") as infile, \
         open(good_csv, "w", encoding="utf-8", newline="") as good_outfile, \
         open(bad_csv, "w", encoding="utf-8", newline="") as bad_outfile:

        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ["sentiment_score"]
        good_writer = csv.DictWriter(good_outfile, fieldnames=fieldnames)
        bad_writer = csv.DictWriter(bad_outfile, fieldnames=fieldnames)
        good_writer.writeheader()
        bad_writer.writeheader()

        for row in reader:
            comment_text = row["comment"]
            sentiment = analyze_sentiment(comment_text)
            row["sentiment_score"] = sentiment
            if sentiment >= 3:
                good_writer.writerow(row)
            else:
                bad_writer.writerow(row)
    print(f"Sentiment analysis completed. Good reviews saved to {good_csv}, bad reviews saved to {bad_csv}.")

def count_reviews(good_csv, bad_csv):
    """
    Count the number of reviews in the good and bad CSV files.
    """
    def count_rows(csv_file):
        with open(csv_file, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            next(reader)
            return sum(1 for _ in reader)
    good_count = count_rows(good_csv)
    bad_count = count_rows(bad_csv)
    print(f"Total Good Reviews: {good_count}")
    print(f"Total Bad Reviews: {bad_count}")
    return good_count, bad_count

def read_reviews(csv_file):
    """
    Read comments from a CSV file and return a single text block.
    (Assumes column 'comment'.)
    """
    reviews = []
    with open(csv_file, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            reviews.append(row.get("comment", "").strip())
    return " ".join(reviews)

######################################
# LLM SUMMARIZATION & VIDEO IDEA FUNCTIONS
######################################

def summarize_good(text, review_type):
    """
    Summarize good reviews using GPT-4o mini.
    """
    prompt = f"""
You are a professional YouTube content strategist. Summarize the following {review_type} reviews.
Highlight key patterns, insights, and audience sentiments.
Provide a structured summary in 3-4 sentences.

**Reviews:**
{text}
    """
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert in content analysis."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def summarize_bad(text, bias_score, misinformation_score, toxic_score, review_type):
    """
    Summarize bad reviews and suggest improvement areas using GPT-4o mini.
    """
    prompt = f"""
You are a professional YouTube content strategist specializing in media integrity analysis.
Summarize the following {review_type} reviews and analyze the following scores:
- Bias Score: {bias_score}
- Misinformation Score: {misinformation_score}
- Toxic Score: {toxic_score}

Provide a clear summary (3-4 sentences), list key areas for improvement, and give actionable recommendations.
    """
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert in YouTube content strategy and media integrity analysis."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def generate_video_ideas(good_summary, bad_summary):
    """
    Generate improved video ideas based on review summaries using GPT-4o mini.
    """
    prompt = f"""
You are a YouTube content strategist. Based on the summaries below, generate 3 innovative video ideas.

**Good Reviews Summary:**
{good_summary}

**Bad Reviews Summary:**
{bad_summary}

Provide each idea with a title and a brief explanation.
    """
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert YouTube strategist."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

######################################
# RAG & MEMORY FUNCTIONS
######################################

def extract_pdf_text(file_path):
    """Extract and return text from a PDF file."""
    text = ""
    with open(file_path, "rb") as f:
        pdf_reader = PyPDF2.PdfReader(f)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def chunk_text(text, chunk_size=200):
    """
    Split text into chunks of approximately chunk_size words.
    """
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Build vector index for RAG using SentenceTransformer and FAISS.
_embedder = SentenceTransformer('all-MiniLM-L6-v2')
_index = None
_chunks = None

def build_faiss_index(chunks):
    """
    Build and return a FAISS index for the given list of text chunks.
    """
    embeddings = _embedder.encode(chunks)
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(np.array(embeddings, dtype=np.float32))
    return index

def retrieve_pdf_context(query, chunks, index, k=3):
    """
    Retrieve the k most relevant text chunks based on the query.
    """
    query_embedding = _embedder.encode([query])
    distances, indices = index.search(np.array(query_embedding, dtype=np.float32), k)
    retrieved = [chunks[idx] for idx in indices[0]]
    return " ".join(retrieved)

# Conversation Memory Functions

conversation_history = []

def update_conversation(role, content):
    """Append a message to the conversation history."""
    conversation_history.append({"role": role, "content": content})

def build_memory_prompt(new_message):
    """
    Build a prompt from the conversation history and the new message.
    """
    history_text = "\n".join(f"{msg['role']}: {msg['content']}" for msg in conversation_history)
    return history_text + f"\nuser: {new_message}\n"
