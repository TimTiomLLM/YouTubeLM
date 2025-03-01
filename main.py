from youtubelm_functions import *

def main():
    # Step 1: Retrieve video data
    video_url = input("Enter YouTube video URL: ")
    video_data = analyze_video(video_url)
    
    print("\n--- Video Data ---")
    print("Video Title:", video_data["title"])
    if video_data["transcript"]:
        print("Transcript (first 500 chars):", video_data["transcript"][:500] + "...")
    else:
        print("Transcript: Not available.")
    print("Number of comments fetched:", len(video_data["comments"]))
    
    # Save and reload comments from CSV
    save_comments_to_csv(video_data["comments"], "video_comments.csv")
    comments_from_csv = load_comments_from_csv("video_comments.csv")
    
    # Step 2: Compute sub-scores
    bias_score = calculate_bias_score(video_data["transcript"])
    misinformation_score = calculate_misinformation_score(video_data["transcript"])
    toxic_score = calculate_toxic_score(video_data["comments"])
    
    print("\n--- Sub-Scores ---")
    print(f"Bias Score: {bias_score:.2f}")
    print(f"Misinformation Score: {misinformation_score:.2f}")
    print(f"Toxic Score: {toxic_score:.2f}")
    
    integrity_score, explanation = calculate_integrity_score(bias_score, misinformation_score, toxic_score)
    print("\n--- Integrity Report ---")
    print("Overall Integrity Score:", integrity_score)
    print("Explanation:", explanation)
    
    # Step 3: Review categorization via sentiment analysis
    categorize_and_save_reviews("video_comments.csv", "good_reviews.csv", "bad_reviews.csv")
    good_count, bad_count = count_reviews("good_reviews.csv", "bad_reviews.csv")
    good_percentage = good_count / (good_count + bad_count)
    print(f"\nGood review percentage: {good_percentage:.2%}")
    
    # Step 4: LLM Summarization and Video Idea Generation
    good_reviews_text = read_reviews("good_reviews.csv")
    bad_reviews_text = read_reviews("bad_reviews.csv")
    
    print("\nSummarizing Good Reviews...")
    good_summary = summarize_good(good_reviews_text, "good")
    
    print("\nSummarizing Bad Reviews...")
    bad_summary = summarize_bad(bad_reviews_text, bias_score, misinformation_score, toxic_score, "bad")
    
    print("\nGenerating Video Ideas...")
    video_ideas = generate_video_ideas(good_summary, bad_summary)
    
    # Print final outputs
    print("\n### Summary of Good Reviews ###")
    print(good_summary)
    
    print("\n### Summary of Bad Reviews ###")
    print(bad_summary)
    
    print("\n### Suggested Video Ideas ###")
    print(video_ideas)

if __name__ == "__main__":
    main()
