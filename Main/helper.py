from gradio_client import Client
import ast
import glob
import os
import shutil
import base64
import streamlit as st
import torch 
import torch.nn as nn
import google.generativeai as genai
import os 
import numpy as np
from gtts import gTTS
import easyocr

models = {
    'MAGNet' : 'https://fffiloni-magnet.hf.space/' , 
    'AudioLDM-2' : 'https://haoheliu-audioldm2-text2audio-text2music.hf.space/' , 
    'Riffusion' : 'https://fffiloni-spectrogram-to-music.hf.space/' , 
    'Mustango' : 'https://declare-lab-mustango.hf.space/' , 
    'MusicGen' : 'https://facebook-musicgen.hf.space/' , 
    'Kosmos' : 'https://kosmos-music.hf.space/' , 
    'MoonDream' : 'https://vikhyatk-moondream1.hf.space/' , 
    'AudioLDM-2' : 'https://haoheliu-audioldm2-text2audio-text2music.hf.space/'
}

def get_image_base64_str(image_path) : 
    '''
    Convert the image to base64 string

    Args :

        1) image_path : str : Path to the image

    Returns :
    
        1) str : Base64 string of the image
    '''

    return base64.b64encode(open(image_path , "rb").read()).decode('utf-8')

def set_bg_hack(main_bg):
    '''
    Set the background of the streamlit app

    Args :
    
        1) main_bg : str : Path to the background image

    Returns :
    
        1) None
    '''


    main_bg_ext = "png"
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

def get_markdown_iamge(image_path) : 
    '''
    Convert the image to markdown format

    Args :

        1) image_path : str : Path to the image

    Returns :
    
        1) str : Markdown string of the image
    '''

    return f'''<img 
    src="data:image/jpeg;base64,{base64.b64encode(open(image_path, "rb").read()).decode()}"  
    width="575" 
    height="200">'''

class NeuralNetwork(nn.Module) :
    '''
    Neural Network for Image Classification
    '''


    def __init__(self) :

        super(NeuralNetwork , self).__init__()

        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(28 * 28 , 128)

        self.relu = nn.ReLU()

        self.linear2 = nn.Linear(128 , 10)

    def forward(self , x) :
        '''
        Forward Pass of the Neural Network

        Args :

            1) x : Tensor : Input Tensor

        Returns :

            1) Tensor : Output Tensor
        '''

        x = self.flatten(x)

        x = self.linear1(x)

        x = self.relu(x)

        x = self.linear2(x)

        return x


def check_api(model) :
    '''
    Check if the API is up and running

    Args :

        1) model : str : Name of the model

    Returns :

        1) Tuple : (Client , str) : Client Object and Status of the API
    ''' 

    try : 
    
        client = Client(models[model])
        return (client , 'API is up and running !')
    
    except : return (None , 'API is down !')

def get_caption(image) : 
    '''
    Get the caption for the image

    Args :

        1) image : str : Path to the image

    Returns :

        1) str : Caption for the image
    '''

    img_to_txt = Client(models['MoonDream'])

    response = img_to_txt.predict(
        'Main/Assets/Generated/Gen.jpg' , 
        'Describe this Image Precisely in Detail' , 
        api_name = '/answer_question')

    return response

def get_music(text) :
    '''
    Get the music for the text

    Args :

        1) text : str : Text to convert to music

    Returns :

        1) str : Path to the music file
    ''' 

    txt_to_audi = Client('https://haoheliu-audioldm2-text2audio-text2music.hf.space/')

    destination_path = 'Main/Assets'
    response = txt_to_audi.predict(
        text , 
        'Low Quality' , 
        10 , 6.5 , 2 , 3 , 
        fn_index = 1)

    directory_path = '/tmp/gradio'

    files = glob.glob(directory_path + '/*')
    files.sort(key=os.path.getmtime)


    sec_files = os.listdir(files[0])
    sec_files = files[0] + '/' + sec_files[0]

    return sec_files

def answer_question(question) :
    '''echo
    Answer the question

    Args :

        1) question : str : Question to answer

    Returns :

        1) str : Answer to the question
    '''

    genai.configure(api_key = st.secrets['API_KEY'])

    resume_text = open('Main/Assets/TextFiles/Resume.txt').read()
    about_me = open('Main/Assets/TextFiles/link_about_me.txt').read()

    model = genai.GenerativeModel('gemini-pro')

    prompt = f'''
    This is my Resume 

    {resume_text}

    This is my about me

    {about_me}

    Now someone else will ask you a question and you have to answer it

    {question}
    '''

    response = model.generate_content(prompt)

    try : return response.text
    except : 
        try : response.parts[0]
        except : return 'Sorry, I cannot answer this question'

def get_text_from_img() : 

    reader = easyocr.Reader(['en'])
    text = reader.readtext('Main/Assets/MedImages/Image.jpg')

    text = [
        val[1]
        for val 
        in text
    ]

    text = ' '.join(text)

    genai.configure(api_key = st.secrets['API_KEY'])
    model = genai.GenerativeModel('gemini-pro')

    prompt = f'''
Consider yourself a doctor. This is a prescription/test report of a Patient
{text}

Analyse the Report and provide suggestions and your analysis as text. If there is a difficult word. Try to explain the word in simple language. Add a disclaimer to suggest the pateint to visit a doctor if necessary. Do not use lines like I am not a doctor and other. Use Patient name wherevar necessary. If telling for a desiase, also tell expected symptons that the person might be feeling 
    '''

    response = model.generate_content(prompt)

    try : return response.text
    except : 
        try : response.parts[0]
        except : return 'Sorry I couldn Understand the Prescription'

def get_speech_from_text(text) : 

    myobj = gTTS(
        text = text , 
        lang = 'en' , 
        slow = False
    ) 

    myobj.save('Main/Assets/MedSpeech/Prescription.mp3')
