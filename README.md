# Lectorsync 1.0

## Overview

**Table of Contents**
1. [Overview](#overview)
2. [Files and Repository](#files-and-repository)
3. [Installation, Usage and Configuration](#installation-usage-and-configuration)
   - [Gradio](#gradio)
   - [Run Locally](#run-locally)
     - [Setup](#setup)
     - [Usage](#usage)
4. [Features and Process](#features-and-process)
   - [Data Creation](#data-creation)
   - [Models](#models)
5. [Data Creation](#data-creation-1)
6. [Transcription](#transcription)
7. [Summariser](#summariser)
8. [Translator](#translator)
9. [Classifier](#classifier)
10. [Gradio](#gradio-1)
11. [Conclusion](#conclusion)
12. [Links](#links)
    - [Datasets](#datasets)
    - [Models](#models-1)
    - [Github Pages](#github-pages)

- Lectorsync is an innovative educational technology tool designed to enhance the accessibility and comprehension of lecture content in multilingual settings. 
- Utilising cutting-edge AI technologies, this platform transcribes spoken lectures into text, and summarises, translates, and classifies the lecture text using custom-trained models.
- This integrated approach provides English-speaking universities in Spain with a powerful tool to instantly convert, simplify, and translate academic lectures, making them accessible and comprehensible to a diverse student body. 
- This project was developed by Alexander Benady, Pablo Chamorro, Gelai Serafico, Sebastián Farje, Gabriel Haftel, and Christian Ranon. 

### Models  
 - **Transcription**: Utilizes the OpenAI Whisper model for accurate transcription. 
 - **Summarization**: Employs a custom-trained Pegasus model, tailored for lecture content.   
 -  **Translation**: Uses a custom-trained Helsinki-NLP models for reliable and coherent translations. 
 - **Classification**:  Uses a custom-trained RoBERTa-based classifier to identify the main academic fields of lectures.
 - Each model is available on huggingface for easy access and use. The links will be provided at the end of this file.  

## Repository

Github: https://github.com/alexanderbenadyieu/lectorsync

Gradio app: https://huggingface.co/spaces/AlexanderBenady/lectorsync

## Files 
Github repository:

     Lectorsync/
    │
    ├── 1_DataCreation/
    │   ├── 1_DataGenerationLectures.ipynb - Notebook for generating detailed academic lecture texts.
    │   ├── 2_DataGenerationSummaries.ipynb - Notebook for generating concise summaries from the lecture texts.
    │   ├── 3_TranslationDataFormatting.ipynb - Prepares translation data for training models.
    │   └── 4_DeepLTranslator.ipynb - Using the DeepL API to translate summaries from English to Spanish.
    │
    ├── 2_Models/
    │   ├── summarizer-stats/ - Directory storing performance statistics and other metrics for the summarization models.
    │   ├── 1_transcriber.ipynb - Notebook containing the model and process for transcribing audio to text.
    │   ├── 2_autotransformer_summarizers.ipynb - Notebook containing training and evaluation of several summarization models.
    │   ├── 3_translator.ipynb - Notebook illustrating the translation model setup and performance.
    │   ├── 4_topic_classification.ipynb - Notebook training and classifying texts into predefined categories.
    │   └── trial_summarizer_t5.ipynb. - Exploration and trial runs using the T5 model for summarization tasks.
    │
    ├── 3_FinalModel/
    │   ├── final_pipeline.py - Script to run the final application, without the gradio implemented, only through CLI
    │   ├── app.py - Gradio version of the final application
    │   ├── requirements.txt - Install for correct libraries   
    │
    ├── data/
    │   ├── Lecture_Topics.xlsx - Excel file containing the list of lecture topics used for generating data.
    │   ├── generated_lectures.csv - CSV file containing generated lectures.
    │   └── summary_translations.csv - CSV file containing the English summaries and their Spanish translations.
    │
    └── README.md


## Installation, Usage and Configuration

- To run the script locally, download final_pipeline.py. The script is compatible with Python 3.x. 
- You will need the following libraries:
```pip install torch transformers pydub```

- If running the gradio version, download app.py and install the following libraries:
```pip install torch transformers pydub gradio sentencepiece```
- Or run this:
```pip install -r requirements.txt```

Additionally, you will need to install ffmpeg to be able to convert mp3s to wav.
- Install homebrew if not available: 
```/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"```
- Install ffmpeg:
```brew install ffmpeg ```

### Run Locally - no Gradio 

1. **Run the Script**:
	Run the file through your command line.
3. **Input File Path**:
When prompted, enter the full path to the text file (.txt), MP3 file (.mp3), or WAV file (.wav) that you want to process. The script can handle text for summarization, translation, and classification tasks, or audio files for transcription followed by any of the aforementioned text processing tasks. 
4. **Select Processing Tasks**: 
	After loading the file, you will be asked to choose which tasks to perform on the input data. Enter the number corresponding to your choice when prompted:
	1.  Summarization
	2.  Translation
	3.  Classification
	4.  Summarization + Translation
	5.  Summarization + Classification
	6.  Translation + Classification
	7.  Summarization + Translation + Classification

- If you choose summarisation (options 4, 5, 7) the translation and classification will be done on the summarised text, not on the transcript. 

### Run the Gradio

To run locally:
- Run the app.py in the terminal
- After running, open http://127.0.0.1:7860/ in a browser

On Hugginface:
- Open https://huggingface.co/spaces/AlexanderBenady/lectorsync


## Process 

## Data Creation

- Due to the lack of existing datasets that fit the unique requirements of our AI models, we chose to create our own synthetic dataset, specifically designed to train our models on the tasks of transcription, summarization, translation, and topic classification.

- We identified five main academic fields to ensure a dataset: Social Sciences, Arts, Natural Sciences, Business and Law, Engineering and Technology. For each field, we selected eight different subjects. Using ChatGPT, we generated 25 lecture titles for each subject across all fields, simulating a university course. In total, this comprised 1,000 unique lecture titles.

- The file with the topics and the prompts used in the following steps can be found in [lectorsync/blob/main/data/Lecture_Topics.xlsx](https://github.com/alexanderbenadyieu/lectorsync/blob/main/data/Lecture_Topics.xlsx).

- The next step was to create the text for each of the lectures. This was carried out in the notebook [lectorsync/blob/main/1_DataCreation/1_DataGenerationLectures.ipynb](https://github.com/alexanderbenadyieu/lectorsync/blob/main/1_DataCreation/1_DataGenerationLectures.ipynb).

- We used the OpenAI API to create a detailed lecture for each of the 1,000 titles. This approach provided a rich source of diverse educational content, simulating actual lecture materials.

- A function named generate_lecture was defined to use GPT-3.5-turbo model to generate, for each lecture, a detailed, engaging, and conversationally toned lecture of at least 1000 words.

- Each lecture was then summarized into a concise summary in the [lectorsync/blob/main/1_DataCreation/2_DataGenerationSummaries.ipynb](https://github.com/alexanderbenadyieu/lectorsync/blob/main/1_DataCreation/2_DataGenerationSummaries.ipynb) notebook.

- generate_summary is defined to generate a summary of a given text using the GPT-4 model. The prompt asks to create a summary limited to 120 words, focusing on the

- The final dataset with the Field, Topic, Lecture Name, Lecture Text and Summary was pushed to HuggingFace: [AlexanderBenady/generated_lectures](https://huggingface.co/datasets/AlexanderBenady/generated_lectures).

- The summaries were then translated into Spanish using the DeepL API, preparing the dataset for training the translation models, as defined in the notebook [lectorsync/blob/main/1_DataCreation/4_DeepLTranslator.ipynb](https://github.com/alexanderbenadyieu/lectorsync/blob/main/1_DataCreation/4_DeepLTranslator.ipynb).

- A new dataset was created with only the summaries and their translation and pushed to HuggingFace: [AlexanderBenady/lecture_summary_translations_english_spanish](https://huggingface.co/datasets/AlexanderBenady/lecture_summary_translations_english_spanish)

- An auxiliary dataframe was compiled with translations from European Commission text from English to Spanish, to be used in Training or Testing if necessary, as detailed in the notebook [lectorsync/blob/main/1_DataCreation/3_TranslationDataFormatting.ipynb](https://github.com/alexanderbenadyieu/lectorsync/blob/main/1_DataCreation/3_TranslationDataFormatting.ipynb), found in [AlexanderBenady/english_spanish_translations](https://huggingface.co/datasets/AlexanderBenady/english_spanish_translations).



## Transcription

This script demonstrates the use of OpenAI's Whisper model for automatic speech recognition (ASR). It utilizes the `transformers` library to apply Whisper Large V3 to process speech data, converting spoken language into text. 

Thee notebook for this part can be found in [lectorsync/blob/main/2_Models/1_transcriber.ipynb](https://github.com/alexanderbenadyieu/lectorsync/blob/main/2_Models/1_transcriber.ipynb)

### Key Components

#### Model and Processor
- **Model**: The script uses the `openai/whisper-large-v3` model from the Hugging Face `transformers` library. This model is specifically tuned for speech recognition tasks.
- **Processor**: An auto processor is used to handle the input data preprocessing and output postprocessing, ensuring the audio data is correctly formatted for the model.

#### Configuration
- **Pipeline**: A speech recognition pipeline is created to streamline the processing of audio files. This pipeline handles everything from feature extraction to decoding the model's predictions.
- **Parameters**:
  - `max_new_tokens`: Limits the generation of tokens to 128, which focuses the model's output.
  - `chunk_length_s`: Processes audio in 30-second chunks to manage memory usage effectively.
  - `batch_size`: Set to 16, allowing multiple samples to be processed simultaneously, enhancing throughput.
  - `return_timestamps`: Disabled to simplify the output, focusing solely on the transcribed text.

#### Execution
- **Sample Audio**: Specifies a path to a sample audio file, which the pipeline uses to generate a transcription.
- **Output**: The result of the speech recognition process is printed, showing the transcribed text of the provided audio sample.

### Usage
To use this script on its own, simply update the `sample` variable with the path to your audio file and run the script. The output will provide the transcribed text of the audio.
In the main.py file, this code is part of the pipeline, where you can upload an .wav file and it will transcribe it for you. 


## Summariser

The Summarizer module in our project is tailored to efficiently condense academic lectures into precise, informative summaries. This document provides an overview of the training process, the models evaluated, and the criteria for selecting the final model used in our tool.

Notebook with initial trial, using t5-small, can be found in [lectorsync/blob/main/2_Models/trial_summarizer_t5.ipynb](https://github.com/alexanderbenadyieu/lectorsync/blob/main/2_Models/trial_summarizer_t5.ipynb)

Notebook with tuning of pegasus, prophet and xlmprophet can be found in [lectorsync/blob/main/2_Models/2_autotransformer_summarizers.ipynb](https://github.com/alexanderbenadyieu/lectorsync/blob/main/2_Models/2_autotransformer_summarizers.ipynb)
  
### Model Training
We used multiple models for training the summarizer, including Pegasus, ProphetNet, XLM-ProphetNet, and MT5.

- **Environment Setup**: Training was facilitated by the AutoTransformers library, which was installed directly from GitHub using:

      ```bash 
      pip install -qU git+https://github.com/lenguajenatural-ai/autotransformers.git

### Evaluation Metric: ROUGE-2

ROUGE-2 was chosen as the primary evaluation metric for summarization quality. It measures the overlap of bigrams between the generated summary and a reference summary, assessing how well the generated text preserves information from the original lecture.

**Model Selection and Comparative Analysis**

Based on our comparative analysis (see Table of Results below), the Pegasus model scores the highest in the ROUGE-2 metric among all tested models.

In addition there good stability across the other ROUGE metrics (ROUGE-1 and ROUGE-L) while maintaining a shorter summary length compared to XLM-PROPHET and PROPHET, but appears to keep key information instead of dropping it like MT5.


| Model       | ROUGE-1  | ROUGE-2  | ROUGE-L  | ROUGE-Lsum | Gen Len  |
|-------------|----------|----------|----------|------------|----------|
| Pegasus     | 43.8758  | 20.1981  | 30.8852  | 40.6891    | 78.6333  |
| XLMPROPHET  | 42.4824  | 18.5202  | 32.1804  | 39.1061    | 128.0000 |
| Prophet     | 40.7825  | 16.0289  | 30.2197  | 37.1005    | 127.6666 |
| MT5         | 22.1939  | 13.1137  | 19.5421  | 20.8396    | 20.0000  |


### Usage
The summarizer can be loaded using the following script:

    from  transformers  import  AutoTokenizer, AutoModelForSeq2SeqLM
    model_name = "cranonieu2021/pegasus-on-lectures" 
    tokenizer  =  AutoTokenizer.from_pretrained(model_name)
    model  =  AutoModelForSeq2SeqLM.from_pretrained(model_name)




## Translator

The translation component of our project utilizes a fine-tuned MarianMT model from the Helsinki-NLP group, specifically tailored for English to Spanish translation. This model is part of the OPUS-MT project and leverages the comprehensive OPUS dataset for training.

Notebook can be found in [lectorsync/blob/main/2_Models/3_translator.ipynb](https://github.com/alexanderbenadyieu/lectorsync/blob/main/2_Models/3_translator.ipynb)

### Model Selection and Training
We chose the `Helsinki-NLP/opus-mt-en-es` model for its robust pre-training on diverse data sources, making it highly effective for general translation tasks. During the project, our NLP professor recommended this model for its proven efficiency and adaptability.

To tailor the model more closely to our project's needs, we conducted additional training on our own dataset, which consists of English to Spanish translations of academic summaries. This approach helped avoid potential overfitting while enhancing the model's performance on academic texts, which differ significantly from the more common governmental and legislative texts found in standard training datasets.

### Training Process
- **Preprocessing**: Text data was preprocessed to ensure compatibility with the model, including tokenization and padding.
- **Hyperparameters**:
  - Learning Rate: 2e-5
  - Batch Size: 16
  - Number of Epochs: 3
  - Weight Decay: 0.01
  - Evaluation Strategy: Epoch

### Usage
The translator can be loaded using the following script:

    from transformers import MarianMTModel, MarianTokenizer 
    
    model_name = "sfarjebespalaia/enestranslatorforsummaries" 
    tokenizer = MarianTokenizer.from_pretrained(model_name) 
    model = MarianMTModel.from_pretrained(model_name)

  


## Classifier

- The classifier is a fine-tuned **RoBerta** model that is trained on our dataset. We have 5 predefined categories, as mentioned earlier: 
	- Social Sciences
	- Arts
	- Natural Sciences
	- Business and Law
	- Engineering and Technology
	
Notebook can be found in [lectorsync/blob/main/2_Models/4_topic_classification.ipynb](https://github.com/alexanderbenadyieu/lectorsync/blob/main/2_Models/4_topic_classification.ipynb)

### Training procedures:
- The dataset was split into 80% training data, 10% validation data and 10% test data. We ensured that each split would have a proportional number of lectures per academic field.
-  Hyperparameter tuning was also carried out. 
	The following hyperparameters were used during training:

	- learning_rate: 2e-05
	- train_batch_size: 32
	- eval_batch_size: 32
	- seed: 42
	- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
	- lr_scheduler_type: linear
	- num_epochs: 2
- Training results:

    |Training Loss|Epoch|Step|Validation Loss|Precision|Recall|F1-score|Accuracy|
    |-------------|-----|----|---------------|---------|------|--------|--------|
    |No log       |1.0  |25  |0.4560         |0.9362   |0.93  |0.9308  |0.93    |
    |No log       |2.0  |50  |0.3287         |0.9519   |0.95  |0.9505  |0.95    |

- It achieves the following results on the test set:

	-  Loss: 0.5266
	-   Precision: 0.9244
	-   Recall: 0.9200
	-   F1-score: 0.9198
	-   Accuracy: 0.92
- This model proved to have higher accuracy rates and lower loss values than other models, such as the SGD Classifier, or other transformer models.


## Conclusion

- Throughout the development of SyncLecture, we encountered several challenges that tested our resourcefulness and technical capabilities. 
    - The main one was the **necessity to create our own synthetic dataset**. This step was crucial as it allowed us to tailor the data for training our models. However, generating this dataset incurred significant costs of usage of API resources.
    - Another limitation was the **reduced computational resources**, which limited which models we could attempt to train, and how intensively we could conduct the training. 
    - We also noted was that the **synthesized lecture texts might not fully mimic the complexity and length of actual university lectures**, which typically last between 1 to 1.5 hours. Consequently, there is a potential discrepancy in how our model would perform with real-life, longer transcripts.

- Looking ahead, there are several avenues for future improvements:
    - **Expanding Language Support**: We aim to include more languages in our translation module, broadening the accessibility and applicability of SyncLecture.
    - **Structured Summaries**: Implementing a notes-style format for summaries could make it more useful for academic purposes.
    - **Utilising Real Data**: Perhaps most critically, transitioning from synthetic to real, annotated lecture data would likely enhance the realism and accuracy of our AI models. 

## Links
Below are the links that we used on Hugging Face:

**Datasets**
- [Lectures](https://huggingface.co/datasets/AlexanderBenady/generated_lectures)
- [Lecture Translations - English to Spanish](https://huggingface.co/datasets/AlexanderBenady/lecture_summary_translations_english_spanish)
- [English to Spanish Translations for Training](https://huggingface.co/datasets/AlexanderBenady/english_spanish_translations)

**Models**
- [Transcriber]()
- [Summarizer](https://huggingface.co/cranonieu2021/pegasus-on-lectures)
- [English to Spanish Translator](https://huggingface.co/sfarjebespalaia/enestranslatorforsummaries)
- [Classifier](https://huggingface.co/gserafico/roberta-base-finetuned-classifier-roberta1)

Below are relevant links to the github pages of the models we used in this project:

- [Whisper Github](https://github.com/openai/whisper)
- [Pegasus Github](https://github.com/google-research/pegasus)
