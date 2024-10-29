from transformers import pipeline,  AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import InferenceClient
import docx2txt

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
translator = pipeline("translation", model=model, tokenizer=tokenizer)

# Function to split text into manageable chunks
def split_text_by_words(text, max_words=10):
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk)
    
    return chunks

# Function to translate text from English to Hindi
def translate(text):
    chunks = split_text_by_words(text)
    translations = []
    
    for chunk in chunks:
        translation = translator(chunk, max_length=200)[0]['translation_text']
        translations.append(translation)
    
    return " ".join(translations)

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
    return summary[0]['summary_text']

if __name__ == "__main__":
    docx_path = './datastore/IndvsSL.docx'
    context = extract_text_from_docx(docx_path)
    summary = summarize_document(context)
    print(translate(summary))