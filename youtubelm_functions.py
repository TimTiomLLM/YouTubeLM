import os
import re
import csv
import openai
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import torch

# Download VADER lexicon if not already available
nltk.download('vader_lexicon', quiet=True)

# Set your API keys here
YOUTUBE_API_KEY = "Insert your YouTube API key here"
OPENAI_API_KEY = "Insert your OpenAI API key here"
openai.api_key = OPENAI_API_KEY

#########################
# Step 1: Data Retrieval
#########################

def extract_video_id(url):
    """
    Extract the video ID from a YouTube URL.
    Supports standard and shortened URLs.
    """
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, url)
    if match:
        return match.group(1)
    else:
        raise ValueError("Invalid YouTube URL provided.")

def get_youtube_service():
    """Create and return a YouTube service object using the API key."""
    return build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

def fetch_video_title(video_id):
    """Retrieve the video title using the YouTube Data API."""
    youtube = get_youtube_service()
    request = youtube.videos().list(part="snippet", id=video_id)
    response = request.execute()
    if response.get("items"):
        return response["items"][0]["snippet"]["title"]
    return None

def fetch_video_comments(video_id, max_results=100):
    """Fetch up to 'max_results' comments for the given video."""
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
    Attempt to retrieve the video transcript.
    If unavailable via the YouTube transcript API, this is where you could integrate a Whisper ASR.
    """
    try:
        transcript_segments = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = " ".join(segment["text"] for segment in transcript_segments)
        return transcript
    except Exception as e:
        print(f"Transcript not available via API: {e}")
        return None

def analyze_video(video_url):
    """
    Given a YouTube video URL, fetch and return the video title, transcript, and comments.
    """
    video_id = extract_video_id(video_url)
    title = fetch_video_title(video_id)
    transcript = fetch_video_transcript(video_id)
    comments = fetch_video_comments(video_id)
    return {
        "video_id": video_id,
        "title": title,
        "transcript": transcript,
        "comments": comments
    }

def save_comments_to_csv(comments, filename="video_comments.csv"):
    """
    Save a list of comment strings to a CSV file with a single column 'comment'.
    """
    with open(filename, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["comment"])
        for comment in comments:
            writer.writerow([comment])
    print(f"Saved {len(comments)} comments to {filename}")

def load_comments_from_csv(filename="video_comments.csv"):
    """
    Load comments from a CSV file and return them as a list of strings.
    Assumes the CSV file has a header with the column 'comment'.
    """
    comments = []
    with open(filename, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            comments.append(row["comment"])
    return comments

#########################
# Scoring System
#########################

# Initialize the bias model
bias_tokenizer = AutoTokenizer.from_pretrained("rinapch/distilbert-media-bias")
bias_model = AutoModelForSequenceClassification.from_pretrained("rinapch/distilbert-media-bias")

def calculate_bias_score(transcript):
    """
    Calculate a bias score for a transcript using a media bias classifier model.
    The bias score is defined as the neutral probability.
    If the transcript is empty, a neutral score of 0.5 is returned.
    """
    if not transcript:
        return 0.5
    inputs = bias_tokenizer(transcript, return_tensors="pt", truncation=True)
    outputs = bias_model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    neutral_probability = probabilities[0][1].item()  # assume index 1 is "Center"
    return max(0, min(1, neutral_probability))

def calculate_misinformation_score(transcript):
    """
    Calculate a misinformation score based on the transcript.
    Here we use a dummy heuristic: count sentences with numerical claims.
    """
    if not transcript:
        return 0.5
    sentences = transcript.split(".")
    claim_sentences = [s for s in sentences if re.search(r"\d+", s)]
    claim_count = len(claim_sentences)
    score = claim_count / (len(sentences) + 1)
    return min(1, score)

def calculate_toxic_score(comments):
    """
    Calculate the average toxicity (non-toxic sentiment) score from the comments using a toxic-comment model.
    Returns a normalized score between 0 and 1.
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
            label = res['label'].lower()
            if label in ['non_toxic', 'non-toxic', 'clean']:
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
    Combine the sub-scores into an overall integrity score (scale 1-10) using a weighted approach.
    Weights:
      - Bias: 40%
      - Misinformation: 40%
      - Toxicity (sentiment): 20%
    """
    weighted = (bias_score * 0.4) + (misinformation_score * 0.4) + (toxic_score * 0.2)
    integrity_score = 1 + (weighted * 9)
    explanation = (f"Weighted calculation: bias={bias_score:.2f} (40%), "
                   f"misinformation={misinformation_score:.2f} (40%), "
                   f"toxicity={toxic_score:.2f} (20%). "
                   f"Final integrity score: {integrity_score:.2f}/10.")
    return integrity_score, explanation

#########################
# Sentiment Analysis & Review Categorization
#########################

# Sentiment analysis model for reviews
sentiment_tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
sentiment_model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

def analyze_sentiment(text):
    """Get sentiment score from text using a BERT-based model (scale 1-5)."""
    inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
    scores = outputs.logits.softmax(dim=1)
    return torch.argmax(scores).item() + 1

def categorize_and_save_reviews(input_csv, good_reviews_csv, bad_reviews_csv):
    """
    Categorize comments from a CSV file into good and bad reviews based on sentiment score.
    Writes two CSV files with an added 'sentiment_score' column.
    """
    with open(input_csv, "r", encoding="utf-8") as infile, \
         open(good_reviews_csv, "w", encoding="utf-8", newline="") as good_outfile, \
         open(bad_reviews_csv, "w", encoding="utf-8", newline="") as bad_outfile:

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
    print(f"Sentiment analysis completed. Good reviews saved to {good_reviews_csv}, bad reviews saved to {bad_reviews_csv}.")

def count_reviews(good_reviews_csv, bad_reviews_csv):
    """Count the number of good and bad reviews from CSV files."""
    def count_rows(csv_file):
        with open(csv_file, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            next(reader)
            return sum(1 for _ in reader)
    good_count = count_rows(good_reviews_csv)
    bad_count = count_rows(bad_reviews_csv)
    print(f"Total Good Reviews: {good_count}")
    print(f"Total Bad Reviews: {bad_count}")
    return good_count, bad_count

#########################
# LLM Summarization & Video Ideas
#########################

def read_reviews(csv_file):
    """
    Read reviews from a CSV file and return as a single text block.
    Assumes the CSV file uses the column 'comment'.
    """
    reviews = []
    with open(csv_file, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            reviews.append(row.get("comment", "").strip())
    return " ".join(reviews)

def summarize_good(text, review_type):
    """Use GPT-4o mini API to summarize good reviews."""
    prompt = f"""
You are a professional YouTube content strategist. Your task is to summarize the following {review_type} reviews.
Identify recurring themes and patterns in the feedback. Highlight what resonates most with the audience.
Provide a clear, structured summary of the main points in 3-4 sentences.

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
    """Use GPT-4o mini API to summarize bad reviews and provide improvement areas."""
    prompt = f"""
You are a professional YouTube content strategist specializing in media integrity analysis.
Your task is to summarize the following {review_type} reviews and analyze the integrity scores.

**Reviews:**
{text}

**Analysis Parameters:**
- Bias Score: {bias_score} (Higher indicates stronger bias)
- Misinformation Score: {misinformation_score} (Higher indicates greater misinformation risk)
- Toxic Score: {toxic_score} (Higher indicates greater toxicity)

Your task:
- Summarize key themes in 3-4 sentences.
- Identify the top area for improving credibility and engagement.

Provide your response in a clear, structured format.
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
    pass