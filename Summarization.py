# Install necessary packages

from huggingface_hub import InferenceClient
from transformers import pipeline
import docx2txt

# Initialize Hugging Face Inßerence Client
client = InferenceClient(api_key="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")

def extract_text_from_docx(docx_path):
    """
    Extracts text from a Word document.

    Parameters:
    docx_path (str): Path to the Word document.

    Returns:
    str: Extracted text from the document.
    """
    return docx2txt.process(docx_path)

def summarize_document(context):
    summarizer =  pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(context, max_length=750, min_length=128, do_sample=False)
    return summary

if __name__ == "__main__":
    docx_path = './datastore/IndvsSL.docx'
    context = extract_text_from_docx(docx_path)
    print(summarize_document(context))
