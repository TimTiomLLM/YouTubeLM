import streamlit as st
import os
import openai
import matplotlib.pyplot as plt
from youtubelm_functions import (
    analyze_video,
    save_comments_to_csv,
    load_comments_from_csv,
    calculate_bias_score,
    calculate_misinformation_score,
    calculate_toxic_score,
    calculate_integrity_score,
    categorize_and_save_reviews,
    count_reviews,
    read_reviews,
    summarize_good,
    summarize_bad,
    generate_video_ideas,
    extract_pdf_text,
    chunk_text,
    build_faiss_index,
    retrieve_pdf_context,
    build_memory_prompt,
    update_conversation
)

# -------------------------------
# Streamlit Page Configuration
# -------------------------------
st.set_page_config(page_title="YouTubeLM", layout="wide")
st.title("YouTubeLM")
st.write("This app analyzes a YouTube video for integrity, review sentiment, and generates a content strategy proposal along with an interactive chatbot.")

# -------------------------------
# Initialize Session State Variables
# -------------------------------
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = {}

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# -------------------------------
# Input Section (Top)
# -------------------------------
video_url = st.text_input("Enter YouTube video URL:")

# Optionally upload the PDF file if not already present.
pdf_path = 'YouTube-Community-Guidelines-August-2018.pdf'
if not os.path.exists(pdf_path):
    pdf_file = st.file_uploader("Upload YouTube Community Guidelines PDF", type=["pdf"])
    if pdf_file is not None:
        with open(pdf_path, "wb") as f:
            f.write(pdf_file.getbuffer())
        st.success(f"Saved {pdf_file.name} locally.")

if st.button("Analyze Video"):
    if not video_url:
        st.error("Please enter a valid YouTube URL.")
    else:
        with st.spinner("Analyzing video..."):
            # STEP 1: Retrieve video data and save comments.
            video_data = analyze_video(video_url)
            save_comments_to_csv(video_data["comments"], "video_comments.csv")
            _ = load_comments_from_csv("video_comments.csv")

            # STEP 2: Compute scoring and integrity.
            bias = calculate_bias_score(video_data["transcript"])
            misinformation = calculate_misinformation_score(video_data["transcript"])
            toxicity = calculate_toxic_score(video_data["comments"])
            integrity, explanation = calculate_integrity_score(bias, misinformation, toxicity)

            # STEP 3: Review sentiment analysis.
            categorize_and_save_reviews("video_comments.csv", "good_reviews.csv", "bad_reviews.csv")
            good_count, bad_count = count_reviews("good_reviews.csv", "bad_reviews.csv")
            total = good_count + bad_count
            good_percentage = (good_count / total) if total > 0 else 0

            good_text = read_reviews("good_reviews.csv")
            bad_text = read_reviews("bad_reviews.csv")
            good_summary = summarize_good(good_text, "good")
            bad_summary = summarize_bad(bad_text, bias, misinformation, toxicity, "bad")
            video_ideas = generate_video_ideas(good_summary, bad_summary)

            # STEP 4: Build the RAG-based initial report.
            if os.path.exists(pdf_path):
                pdf_text = extract_pdf_text(pdf_path)
                pdf_chunks = chunk_text(pdf_text, chunk_size=200)
                transcript_chunks = chunk_text(video_data["transcript"], chunk_size=200) if video_data["transcript"] else []
                all_chunks = transcript_chunks + pdf_chunks
                index = build_faiss_index(all_chunks)
                user_query = (
                    "Combine all the information to generate innovative video ideas while critically reviewing "
                    "the existing content, pointing out problems, and providing specific suggestions for improvement."
                )
                retrieved_context = retrieve_pdf_context(user_query, all_chunks, index, k=5)

                initial_rag_input = f"""
【Video Ideas】:
{video_ideas}

Based on the above information and the following context:
{retrieved_context}

Please provide a detailed and comprehensive proposal with innovative ideas for the next video.
"""
                initial_response = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a professional YouTube content strategist."},
                        {"role": "user", "content": initial_rag_input}
                    ]
                )
                initial_report = initial_response.choices[0].message.content
            else:
                initial_report = "PDF not available, so no initial report generated."

            # Store analysis results in session state.
            st.session_state.analysis_results = {
                "video_data": video_data,
                "bias": bias,
                "misinformation": misinformation,
                "toxicity": toxicity,
                "integrity": integrity,
                "explanation": explanation,
                "good_percentage": good_percentage,
                "good_count": good_count,
                "bad_count": bad_count,
                "good_summary": good_summary,
                "bad_summary": bad_summary,
                "video_ideas": video_ideas,
                "initial_report": initial_report
            }
            st.session_state.analysis_done = True

            # Initialize conversation history with system prompt and initial report.
            system_message = (
                "You are a professional YouTube content strategist familiar with YouTube review rules and creation guidelines.\n\n"
                "Please elaborate on the proposal if the YouTuber has any questions. Use the following information from previous feedback:\n"
                "【Video Ideas】: " + video_ideas + "\n\n"
                "Ensure your response follows the youtube community guidelines."
            )
            st.session_state.conversation_history = [
                {"role": "system", "content": system_message},
                {"role": "assistant", "content": initial_report}
            ]
            st.success("Video analysis completed!")

# -------------------------------
# Layout: Two Columns (Left: Analysis, Right: Chatbot)
# -------------------------------
col1, col2 = st.columns(2)

# Left column: Updated Analysis.
with col1:
    st.header("Video Analysis")
    if st.session_state.analysis_done:
        video_data = st.session_state.analysis_results["video_data"]
        st.markdown(f"**Title:** {video_data['title']}")
        
        st.subheader("Integrity-Scores")
        st.markdown(f"**Bias Score:** {st.session_state.analysis_results['bias']:.2f} / 1")
        st.markdown(f"**Misinformation Score:** {st.session_state.analysis_results['misinformation']:.2f} / 1")
        st.markdown(f"**Toxic Score:** {st.session_state.analysis_results['toxicity']:.2f} / 1")
        st.markdown(f"**Overall Integrity Score:** {st.session_state.analysis_results['integrity']:.2f} / 10")
        st.markdown(f"**Explanation:** {st.session_state.analysis_results['explanation']}")
        
        # Pie chart for overall integrity score.
        integrity_score = st.session_state.analysis_results['integrity']
        integrity_remaining = 10 - integrity_score if integrity_score <= 100 else 0
        fig_int, ax_int = plt.subplots()
        ax_int.pie([integrity_score, integrity_remaining],
                   labels=["Integrity", "Lack of Integrity"],
                   autopct="%1.1f%%", startangle=90)
        ax_int.set_title("Overall Integrity")
        
        # Pie chart for review sentiment (Good vs Bad reviews).
        good_count = st.session_state.analysis_results["good_count"]
        bad_count = st.session_state.analysis_results["bad_count"]
        total_reviews = good_count + bad_count
        if total_reviews > 0:
            fig_review, ax_review = plt.subplots()
            ax_review.pie([good_count, bad_count],
                          labels=["Positive Reviews", "Negative Reviews"],
                          autopct="%1.1f%%", startangle=90)
            ax_review.set_title("Review Sentiment")
        else:
            fig_review = None
        
        # Place the two pie charts side by side.
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.pyplot(fig_int)
        with chart_col2:
            if fig_review:
                st.pyplot(fig_review)
            else:
                st.write("No review data available.")
    else:
        st.info("No analysis data yet. Please analyze a video first.")

with col2:
    st.header("YouTubeLM Chatbot")
    st.markdown(
        """
        <style>
        .chat-history {
            height: 500px;
            overflow-y: auto;
            border: 1px solid #ccc;
            padding: 10px;
            margin-bottom: 10px;
        }
        .sticky-input {
            position: sticky;
            bottom: 0;
            background: white;
            padding: 10px;
            border-top: 1px solid #ccc;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Create an empty container for the chat history.
    chat_container = st.empty()

    def render_chat_history():
        chat_html = '<div class="chat-history">'
        for msg in st.session_state.conversation_history:
            # Skip system messages from being displayed.
            if msg["role"] == "system":
                continue
            if msg["role"] == "assistant":
                chat_html += '<p><strong>YouTubeLM:</strong>' + '\n\n' + msg["content"] + '</p>'
            else:
                chat_html += '<p><strong>' + msg["role"].capitalize() + ':</strong> ' + msg["content"] + '</p>'
        chat_html += '</div>'
        return chat_html

    # Render the current chat history.
    chat_container.markdown(render_chat_history(), unsafe_allow_html=True)

    # Sticky input box at the bottom.
    st.markdown('<div class="sticky-input">', unsafe_allow_html=True)
    user_input = st.text_input("Your question:", key="chat_input")
    if st.button("Send Question", key="send_question"):
        if user_input.strip().lower() in ["exit", "quit"]:
            st.write("Dialogue terminated.")
        elif user_input.strip() != "":
            st.session_state.conversation_history.append({"role": "user", "content": user_input})
            memory_prompt = build_memory_prompt(user_input)
            chat_response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": st.session_state.conversation_history[0]["content"]},
                    {"role": "user", "content": memory_prompt}
                ]
            )
            output = chat_response.choices[0].message.content
            st.session_state.conversation_history.append({"role": "assistant", "content": output})
            chat_container.markdown(render_chat_history(), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
