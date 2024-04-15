python
Copy code
import logging
import os
import warnings
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSpeechSeq2Seq, 
    MarianMTModel, MarianTokenizer, AutoModelForSequenceClassification, 
    AutoProcessor, pipeline
)
import torch
from pydub import AudioSegment

# Suppress user warnings from libraries
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set device and data type based on CUDA availability
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Preload models and tokenizers for summarization
summarizer_tokenizer = AutoTokenizer.from_pretrained('cranonieu2021/pegasus-on-lectures')
summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(
    "cranonieu2021/pegasus-on-lectures", torch_dtype=torch_dtype).to(device)

# Preload models and tokenizers for translation
translator_tokenizer = MarianTokenizer.from_pretrained("sfarjebespalaia/enestranslatorforsummaries")
translator_model = MarianMTModel.from_pretrained(
    "sfarjebespalaia/enestranslatorforsummaries", torch_dtype=torch_dtype).to(device)

# Preload models and tokenizers for classification
classifier_tokenizer = AutoTokenizer.from_pretrained("gserafico/roberta-base-finetuned-classifier-roberta1")
classifier_model = AutoModelForSequenceClassification.from_pretrained(
    "gserafico/roberta-base-finetuned-classifier-roberta1", torch_dtype=torch_dtype).to(device)


def convert_mp3_to_wav(mp3_file_path):
    """
    Converts an MP3 file to WAV format.

    Parameters:
    mp3_file_path (str): Path to the MP3 file.

    Returns:
    str: Path to the created WAV file.
    """
    try:
        wav_file_path = mp3_file_path.replace(".mp3", ".wav")
        audio = AudioSegment.from_mp3(mp3_file_path)
        audio.export(wav_file_path, format="wav")
        logging.info("MP3 file converted to WAV.")
        return wav_file_path
    except Exception as e:
        logging.error(f"Error converting MP3 to WAV: {e}")
        raise

def transcribe_audio(audio_file_path):
    """
    Transcribes audio from a WAV file using a speech-to-text model.

    Parameters:
    audio_file_path (str): Path to the WAV file.

    Returns:
    str: Transcribed text.
    """
    try:
        model_id = "openai/whisper-large-v3"
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, use_safetensors=True)
        model.to(device)
        processor = AutoProcessor.from_pretrained(model_id)
        pipe = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor, device=device)
        result = pipe(audio_file_path)
        logging.info("Audio transcription completed successfully.")
        return result['text']
    except Exception as e:
        logging.error(f"Error transcribing audio: {e}")
        raise

def load_and_process_input(file_path):
    """
    Loads and processes an input file based on its extension.

    Parameters:
    file_path (str): Path to the file.

    Returns:
    str: Processed file content or transcription.
    """
    extension = os.path.splitext(file_path)[-1].lower()
    try:
        if extension == ".txt":
            with open(file_path, 'r') as file:
                return file.read()
        elif extension == ".mp3":
            wav_file_path = convert_mp3_to_wav(file_path)
            return transcribe_audio(wav_file_path)
        elif extension == ".wav":
            return transcribe_audio(file_path)
        else:
            raise ValueError("Unsupported file type provided.")
    except Exception as e:
        logging.error(f"Error processing input file: {e}")
        raise

def process_text(text, summarization=False, translation=False, classification=False):
    """
    Processes text for summarization, translation, and classification based on flags set.

    Parameters:
    text (str): Input text to process.
    summarization (bool): Flag to trigger summarization.
    translation (bool): Flag to trigger translation.
    classification (bool): Flag to trigger classification.

    Returns:
    dict: A dictionary with keys for each processing type that was performed.
    """
    results = {}
    intermediate_text = text  # Start with the original text for processing

    # Summary generation
    if summarization:
        inputs = summarizer_tokenizer(intermediate_text, max_length=1024, return_tensors="pt", truncation=True)
        summary_ids = summarizer_model.generate(inputs.input_ids, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary_text = summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        results['summarized_text'] = summary_text
        intermediate_text = summary_text  # Update intermediate text if summary is used for further processing

    # Text translation
    if translation:
        tokenized_text = translator_tokenizer.prepare_seq2seq_batch([intermediate_text], return_tensors="pt")
        translated = translator_model.generate(**tokenized_text)
        translated_text = ' '.join(translator_tokenizer.decode(t, skip_special_tokens=True) for t in translated)
        results['translated_text'] = translated_text.strip()

    # Text classification
    if classification:
        inputs = classifier_tokenizer(intermediate_text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = classifier_model(**inputs)
        predicted_class_idx = torch.argmax(outputs.logits, dim=1).item()
        labels = {
            0: 'Social Sciences',
            1: 'Arts',
            2: 'Natural Sciences',
            3: 'Business and Law',
            4: 'Engineering and Technology'
        }
        results['classification_result'] = labels[predicted_class_idx]

    return results

def display_results(results):
    """
    Displays the results of the text processing.

    Parameters:
    results (dict): Dictionary containing results from processing functions.
    """
    if 'summarized_text' in results:
        print("Summarized Text:")
        print(results['summarized_text'])
    if 'translated_text' in results:
        print("Translated Text:")
        print(results['translated_text'])
    if 'classification_result' in results:
        print('Classification Result:')
        print(f"This text is classified under: {results['classification_result']}")

def main():
    """
    Main function to run the script. Handles user input and calls other functions.
    """
    print("Loading models, please wait...")

    file_path = input("Enter the path to your text, mp3, or wav file: ")
    if not os.path.isfile(file_path):
        print("File does not exist. Please enter a valid file path.")
        return

    text = load_and_process_input(file_path)

    print("Choose the tasks to perform:")
    print("1. Summarization")
    print("2. Translation")
    print("3. Classification")
    print("4. Summarization + Translation")
    print("5. Summarization + Classification")
    print("6. Translation + Classification")
    print("7. Summarization + Translation + Classification")

    while True:
        try:
            choice = int(input("Please choose your option -> "))
            if choice not in range(1, 8):
                raise ValueError("Please select a valid option from 1 to 7.")
            break
        except ValueError as e:
            print(f"Invalid input: {e}. Please try again.")

    summarization = choice in [1, 4, 5, 7]
    translation = choice in [2, 4, 6, 7]
    classification = choice in [3, 5, 6, 7]

    results = process_text(text, summarization=summarization, translation=translation, classification=classification)
    display_results(results)

if __name__ == "__main__":
    main()
