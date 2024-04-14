import logging
import os
import warnings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSpeechSeq2Seq, MarianMTModel, MarianTokenizer, AutoModelForSequenceClassification, AutoProcessor, pipeline
import torch
from pydub import AudioSegment
import gradio as gr

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.")
warnings.filterwarnings("ignore", message="Due to a bug fix in https://github.com/huggingface/transformers/pull/28687 transcription using a multilingual Whisper will default to language detection followed by transcription instead of translation to English.This might be a breaking change for your use case. If you want to instead always translate your audio to English, make sure to pass `language='en'.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Preload models globally

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load all necessary models and tokenizers
summarizer_tokenizer = AutoTokenizer.from_pretrained('cranonieu2021/pegasus-on-lectures')
summarizer_model = AutoModelForSeq2SeqLM.from_pretrained("cranonieu2021/pegasus-on-lectures", torch_dtype=torch_dtype).to(device)

translator_tokenizer = MarianTokenizer.from_pretrained("sfarjebespalaia/enestranslatorforsummaries")
translator_model = MarianMTModel.from_pretrained("sfarjebespalaia/enestranslatorforsummaries", torch_dtype=torch_dtype).to(device)

classifier_tokenizer = AutoTokenizer.from_pretrained("gserafico/roberta-base-finetuned-classifier-roberta1")
classifier_model = AutoModelForSequenceClassification.from_pretrained("gserafico/roberta-base-finetuned-classifier-roberta1", torch_dtype=torch_dtype).to(device)


def transcribe_audio(audio_file_path):
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

def load_and_process_input(file_info):
    file_path = file_info  # Assuming it's just the path
    original_filename = os.path.basename(file_path)  # Extract filename from path if needed

    extension = os.path.splitext(original_filename)[-1].lower()
    try:
        if extension == ".txt":
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
        elif extension in [".mp3", ".wav"]:
            if extension == ".mp3":
                file_path = convert_mp3_to_wav(file_path)
            text = transcribe_audio(file_path)
        else:
            raise ValueError("Unsupported file type provided.")
    except Exception as e:
        logging.error(f"Error processing input file: {e}")
        raise
    return text



# Ensure the convert_mp3_to_wav accepts and handles a file path correctly
def convert_mp3_to_wav(file_path):
    try:
        wav_file_path = file_path.replace(".mp3", ".wav")
        audio = AudioSegment.from_file(file_path, format='mp3')
        audio.export(wav_file_path, format="wav")
        logging.info("MP3 file converted to WAV.")
        return wav_file_path
    except Exception as e:
        logging.error(f"Error converting MP3 to WAV: {e}")
        raise

def process_text(text, summarization=False, translation=False, classification=False):
    results = {}
    intermediate_text = text  # This will hold either the original text or the summary

    if summarization:
        # Perform summarization
        inputs = summarizer_tokenizer(intermediate_text, max_length=1024, return_tensors="pt", truncation=True)
        summary_ids = summarizer_model.generate(inputs.input_ids, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary_text = summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        results['summarized_text'] = summary_text
        intermediate_text = summary_text  # Update intermediate text to be the summary for further processing

    if translation:
        # Translate the intermediate text (which could be either the original text or the summary)
        tokenized_text = translator_tokenizer.prepare_seq2seq_batch([intermediate_text], return_tensors="pt")
        translated = translator_model.generate(**tokenized_text)
        translated_text = ' '.join(translator_tokenizer.decode(t, skip_special_tokens=True) for t in translated)
        results['translated_text'] = translated_text.strip()

    if classification:
        # Classify the intermediate text (which could be either the original text or the summary)
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

def wrap_process_file(file_obj, tasks):
        if file_obj is None:
            return "Please upload a file to proceed.", "", "", ""

        # Assuming file_obj is a tuple containing (temp file path, original file name)
        text = load_and_process_input(file_obj)
        results = process_text(text, 'Summarization' in tasks, 'Translation' in tasks, 'Classification' in tasks)

        return (results.get('summarized_text', ''),
                results.get('translated_text', ''),
                results.get('classification_result', ''))

def create_gradio_interface():
    with gr.Blocks(theme="huggingface") as demo:
        gr.Markdown("# LectorSync 1.0")
        gr.Markdown("## Upload your file and select the tasks:")
        with gr.Row():
            file_input = gr.File(label="Upload your text, mp3, or wav file")
            task_choice = gr.CheckboxGroup(["Summarization", "Translation", "Classification"], label="Select Tasks")
            submit_button = gr.Button("Process")
        output_summary = gr.Text(label="Summarized Text")
        output_translation = gr.Text(label="Translated Text")
        output_classification = gr.Text(label="Classification Result")

        submit_button.click(
            fn=wrap_process_file,
            inputs=[file_input, task_choice],
            outputs=[output_summary, output_translation, output_classification]
        )

    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch()
