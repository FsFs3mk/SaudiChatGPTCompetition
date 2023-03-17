import whisper
import gradio as gr
import time
from pyChatGPT import ChatGPT
import requests
import json


# Enter your session token here!

secret_token = "zqt_nmmeFhClTmRlpVM82XK_BxfkKYwHn6WX3AJIIQ"
model = whisper.load_model("small")


def gpt_predict(text):
    url = 'https://experimental.willow.vectara.io/v1/completions'
    headers = {'Content-Type': 'application/json', 'customer-id': '2657721878',
               'x-api-key': 'zqt_nmmeFhClTmRlpVM82XK_BxfkKYwHn6WX3AJIIQ'}
    body = {'model': 'text-davinci-003', 'prompt': f"Summrize this: {text}",
            'max_tokens': 20, 'temperature': 0}

    x = requests.post(url, headers=headers, json=body)
    return x.text


def transcribe(audio):

    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)

    # decode the audio
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)
    result_text = result.text

    # Pass the generated text to Audio
    text = gpt_predict(result_text)
    out_result = text

    return [result_text, out_result]


output_1 = gr.Textbox(label="Speech to Text")
output_2 = gr.Textbox(label="Summarized text")


gr.Interface(
    title='OpenAI Whisper and ChatGPT ASR Gradio Web UI',
    fn=transcribe,
    inputs=[
        gr.inputs.Audio(source="microphone", type="filepath")
    ],

    outputs=[
        output_1,  output_2
    ],
    live=True).launch(share=True)
