import streamlit as st
import requests
import time
import os
import streamlit as st

st.set_page_config(page_title="Persona-Driven Document Intelligence", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(to right, #1e0502, #721704, #2d0b04);
        color: #e5e7eb;
    }
.stSlider > div[data-baseweb="slider"] {
    width: 300px !important;
}
    .stTextInput>div>div>input,
    .stFileUploader>div>div>div>input {
        background-color: #2a2b2b;
        color: #e5e7eb;
        border: 1px solid #b30c01;
        border-radius: 0.5rem;
        padding: 10px;
    }

    .stButton>button {
        background-color: #b30c01;
        color: white;
        padding: 10px 20px;
        border-radius: 0.5rem;
        transition: background-color 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #8c0a00;
    }

    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #c4c4c4;
    }

    .stDownloadButton>button {
        background-color: #eb1100;
        color: white;
        border-radius: 0.375rem;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }

    .stDownloadButton>button:hover {
        background-color: #bf1002;
    }

    .summary-section {
        background-color: #1b1b1b;
        padding: 20px;
        border-radius: 0.75rem;
        margin-bottom: 20px;
        border-left: 5px solid #b30c01;
    }

    .section-title {
        color: #c2c2c2;
        font-size: 20px;
        margin-bottom: 10px;
    }

    .section-content {
        color: #ffffff;
        font-size: 16px;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

st.title("Persona-Driven Document Intelligence")
st.markdown("Connecting What Matters - For The User Who Matters")
def format_section(title, content):
    """Format a section of the summary with consistent styling"""
    return f"""
    <div class="summary-section">
        <div class="section-title">{title}</div>
        <div class="section-content">{content}</div>
    </div>
    """

uploaded_file = st.file_uploader("Upload a local PDF", type="pdf", key="pdf_uploader")

persona =  st.radio("choose one option ",
         ["student","teacher","researcher","user"],
         index=None,)
word_limit = st.slider("pick the word limit ",150,300)
status_placeholder = st.empty()
if uploaded_file is not None:
    collections_dir = os.path.join(os.path.dirname(__file__), "Collections","PDFs")
    os.makedirs(collections_dir, exist_ok=True)

    base_filename = uploaded_file.name
    if not base_filename.lower().endswith(".pdf"):
        base_filename += ".pdf"

    collections_path = os.path.join(collections_dir, base_filename)

    if not os.path.exists(collections_path):
        with open(collections_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        st.success(f"✅ File saved to Collections as: `{base_filename}`")
    else:
        st.warning(f"⚠️ File `{base_filename}` already exists in Collections. Skipping save.")


if st.button("Analyze PDF"):
    if uploaded_file :
        with st.spinner("Processing..."):
            status_placeholder.info("Uploading and analyzing the document...")

            try:
                # Send the PDF and input.json as separate file fields
                files = {
                    "file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf"),
                }
                data = {
                    "persona":persona,
                    "word_limit":word_limit,
                }
                response = requests.post(
                    "http://localhost:8000/summarize_local/",
                    files=files,
                    data=data,
                    timeout=3600
                )
                if response.status_code == 200:
                    data = response.json()
                    if "error" in data:
                        status_placeholder.error(f"{data['error']}")
                    else:
                        summary = data.get("summary", "No summary generated.")
                        status_placeholder.success("Summary Ready!")
                        st.text_area("📜 Summary", summary, height=400)
                        st.download_button("⬇️ Download Summary", summary, "summary.md")
                else:
                    status_placeholder.error("Server error.")
            except Exception as e:
                status_placeholder.error(f"Error: {str(e)}")
    else:
        status_placeholder.warning("Please upload both a PDF file and an input.json file.")


st.markdown("---")
