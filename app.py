import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import fitz  # type: ignore # PyMuPDF for PDF processing
from docx import Document  # type: ignore # For DOCX processing
import os

# Ensure NLTK data path is set and download necessary resources
nltk.data.path.append(os.path.join(os.path.expanduser("~"), "nltk_data"))
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in pdf_reader:
        text += page.get_text("text")
    pdf_reader.close()
    return text.strip()

# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = Document(file)
    return " ".join([paragraph.text.strip() for paragraph in doc.paragraphs])

# Preprocess text
def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)

    text = text.replace('\n', ' ').replace('\t', ' ')
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    
    return ' '.join(filtered_tokens)

# Calculate similarity
def calculate_similarity(resume_text, job_description):
    resume_cleaned = preprocess_text(resume_text)
    job_description_cleaned = preprocess_text(job_description)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_cleaned, job_description_cleaned])
    similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    
    return similarity_score[0][0]

# Streamlit app
# Add an image to the app
st.image('zyberfox.jpg', caption='Automated Resume Screening System', use_column_width=True)
def main():
    st.title("Automated Resume Screening System")
    st.header("Compare Multiple Resumes with a Job Description")

    # Upload multiple resumes
    uploaded_files = st.file_uploader("Upload up to 5 resumes (PDF or DOCX):", type=['pdf', 'docx'], accept_multiple_files=True)

    # Input job description
    job_description = st.text_area("Enter the job description:")

    if st.button("Calculate Match"):
        if uploaded_files and job_description:
            similarity_scores = {}
            for file in uploaded_files:
                try:
                    # Extract text based on file type
                    if file.type == "application/pdf":
                        resume_text = extract_text_from_pdf(file)
                    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                        resume_text = extract_text_from_docx(file)
                    else:
                        st.error(f"Unsupported file type for {file.name}.")
                        continue

                    # Calculate similarity
                    score = calculate_similarity(resume_text, job_description)
                    similarity_scores[file.name] = round(score * 100, 2)  # Store as percentage

                except Exception as e:
                    st.error(f"Error processing {file.name}: {e}")

            # Display results if any files were processed
            if similarity_scores:
                st.success("Similarity scores calculated successfully!")
                st.subheader("Similarity Scores (Percentage Match)")

                # Display bar chart
                st.bar_chart(similarity_scores)

                # Display detailed results
                for filename, score in similarity_scores.items():
                    st.write(f"**{filename}**: {score:.2f}%")
        else:
            st.warning("Please upload resumes and enter a job description.")

if __name__ == "__main__":
    main()
