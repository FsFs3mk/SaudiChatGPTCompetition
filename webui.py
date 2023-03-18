import gradio as gr
import whisper
import langs
import requests
import json

lang_list = sorted(langs.LANGUAGES.values())

def transcribe(task, device, language, model_size, mic, file):
        if device =='gpu':
            device = 'cuda'
        args = {'task': task}
        if (model_size == 'tiny.en') or (model_size == 'base.en') or (model_size == 'small.en') or (model_size == 'medium.en'):
            args['language'] = 'english'
        elif (language == 'Detect'):
            args['language'] = None
        else:
            args['language'] = language
        model = whisper.load_model(model_size, device)
        if mic is not None:
            audio = mic
        elif file is not None:
            audio = file
        else:
            return "You must either provide a mic recording or a file"
        trascript = model.transcribe(audio, **args)
        url = "" #insert api url here

        payload = json.dumps({
        "model": "text-davinci-003",
        "prompt": f"Summarize this : {trascript}",
        "max_tokens": 3000,
        "temperature": 0})
        headers = {
        'Content-Type': 'application/json',
        'customer-id': '',#insert customer-id here
        'x-api-key': ''}#insert api key here
        response = requests.request("POST", url, headers=headers, data=payload)
        print(response)
        return {"text":trascript["text"],"summary":response}



demo = gr.Interface(transcribe, 
    inputs=[
        gr.Radio(['transcribe', 'summarize'], label= 'Task'),
        gr.Radio(['gpu', 'cpu'], label= 'Device'),
        gr.Dropdown(lang_list, value='Detect',  label='Audio Language'),
        gr.Dropdown(['tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large'], value='small', label='Model Size'), 
        gr.Audio(label='Microphone Recording', source='microphone', type='filepath'), 
        gr.Audio(source='upload', type='filepath', optional=True, label='Audio File')
        ], 
    outputs="text")
demo.launch(share=True) 
