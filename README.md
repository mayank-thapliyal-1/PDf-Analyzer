
Adobe India Hackathon - Byte Flair (Round 1B)

An offline LLM-powered system for Persona-Driven Document Intelligence

🛠️ What We Built

In Round 1B, we developed a persona-driven document intelligence system that extracts and ranks the most relevant sections and sub-sections from a collection of PDFs. The pipeline adapts to varying **personas** and **job-to-be-done** inputs using prompt engineering and context-aware segmentation.

We leveraged **Gemma 3:1b**, an efficient open-source LLM that runs entirely **offline**, complies with the **<1GB size constraint**, and requires **CPU-only execution**. It processes 3–5 documents in approximately **4 minutes**, which slightly exceeds ideal expectations, we kindly ask you to bear with the additional processing time in favor of accuracy and model richness.

The system produces a structured output JSON containing metadata, ranked sections, and refined sub-section text, ensuring alignment with the competition’s goals of semantic relevance and explainability. All documents in `Collections/input.json` are processed and output to `Collections/output.json`, with no web calls, making it secure, reproducible, and compatible with the evaluation environment. The output after processing is produced in a relevant JSON format printed on the Front-end.


_____________________________________________________________________________________________________________________________________________________________________________________________________________________

Installation & Setup


1. Clone the Repository

```
git clone https://github.com/your-username/Adobe-India-Hackathon-Byte-Flair-1B.git
````



2. Install Dependencies

```
pip install -r requirements.txt
```



3. Install Ollama & Gemma 3 LLM

▪️ Install Ollama (MacOS/Linux)


```bash
curl -fsSL https://ollama.com/install.sh | sh
```

If not supporting your OS, install Ollama from the website - https://ollama.com/download



Download Gemma 3 Model

```
ollama pull gemma3:1b
```



4. Start the Backend (FastAPI)


```
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```




5. Start the Frontend (Streamlit)

```
streamlit run frontend.py
```

To Run On Docker

```
docker build -t ollama-app .
docker run -p 8000:8000 -p 8501:8501 ollama-app
```

  <img width="400" height="600" alt="Gemma 3_1B Robot Illustration" src="https://github.com/user-attachments/assets/e1ef7322-379f-4829-babf-af97cd0494ac" />

