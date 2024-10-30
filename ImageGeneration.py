from transformers import pipeline,  AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import InferenceClient
import docx2txt
from gtts import gTTS
from openai import OpenAI


def image_generation(context):
    client = OpenAI(api_key="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")
    summary = summarize_document(context)
    response = client.images.generate(
        prompt=summary[:64],
        n=1,
        size="1024x1024"
    )
    image_url = response.data[0].url
    print(image_url)

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
    summary = summarizer(context, max_length=100, min_length=64, do_sample=False)
    return summary[0]['summary_text']

if __name__ == "__main__":
    docx_path = './datastore/IndvsSL.docx'
    context = extract_text_from_docx(docx_path)
    image_generation(context)