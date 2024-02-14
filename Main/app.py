import streamlit as st
from copy import deepcopy
from PIL import Image
import torch
import os 

import streamlit.components.v1 as components
from gradio_client import Client

from helper import (
    check_api , 
    get_caption , 
    get_music , 
    get_image_base64_str ,
    set_bg_hack , 
    get_markdown_iamge , 
    NeuralNetwork
)

import torch 
from torchvision.utils import save_image
from presidio_analyzer import AnalyzerEngine

# st.set_page_config(layout = 'wide')

themes = {
    '0' : [
        'Main/Assets/Background/Road.jpg' , 
        '# Road - Where the journey never ends and so does the traffic.' , 
        'If you dont like the Road theme, you can change it by clicking on the `Generate Images` Button' , 
        'Main/Assets/Project Tiles/ML_DL_AI/Road/AI Gen-Text Detection.png' , 
        'Main/Assets/Project Tiles/ML_DL_AI/Road/PII Detection.png' , 
        'Main/Assets/Project Tiles/ML_DL_AI/Road/HUBMAP.png' , 
        'Main/Assets/Project Tiles/ML_DL_AI/Road/LLM Exam.png' , 
        'Main/Assets/Project Tiles/Research/Road/VAl_rind.png' , 
        'Main/Assets/Project Tiles/Development/Road/VA_dev.png'
        ] ,
    '1' : [
        'Main/Assets/Background/Nature.jpg' , 
        '# No, you cannot eat that Mushroom' , 
        'If you dont like the Nature theme, you can change it by clicking on the `Generate Images` Button' ,
        'Main/Assets/Project Tiles/ML_DL_AI/Nature/AI Gen-Text Detection.png' , 
        'Main/Assets/Project Tiles/ML_DL_AI/Nature/PII Detection.png' , 
        'Main/Assets/Project Tiles/ML_DL_AI/Nature/HUBMAP.png' ,
        'Main/Assets/Project Tiles/ML_DL_AI/Nature/LLM Exam.png' ,
        'Main/Assets/Project Tiles/Research/Nature/VAl_rind.png' ,
        'Main/Assets/Project Tiles/Development/Nature/VA_dev.png'
        ] , 
    '2' : [
        'Main/Assets/Background/City.jpg' , 
        '# City - Where the lights never go out and so does the noise.' , 
        'If you dont like the City theme, you can change it by clicking on the `Generate Images` Button' ,  
        'Main/Assets/Project Tiles/ML_DL_AI/City/AI Gen-Text Detection.png' ,
        'Main/Assets/Project Tiles/ML_DL_AI/City/PII Detection.png' ,
        'Main/Assets/Project Tiles/ML_DL_AI/City/HUBMAP.png' ,
        'Main/Assets/Project Tiles/ML_DL_AI/City/LLM Exam.png' ,
        'Main/Assets/Project Tiles/Research/City/VAl_rind.png' ,
        'Main/Assets/Project Tiles/Development/City/VA_dev.png'
        ] ,
    '3' : [
        'Main/Assets/Background/Space.jpg' , 
        '# No, you cannot pee there'  ,  
        'If you dont like the Space theme, you can change it by clicking on the `Generate Images` Button' , 
        'Main/Assets/Project Tiles/ML_DL_AI/Space/AI Gen-Text Detection.png' , 
        'Main/Assets/Project Tiles/ML_DL_AI/Space/PII Detection.png' , 
        'Main/Assets/Project Tiles/ML_DL_AI/Space/HUBMAP.png' ,
        'Main/Assets/Project Tiles/ML_DL_AI/Space/LLM Exam.png' ,
        'Main/Assets/Project Tiles/Research/Space/VAl_rind.png' ,
        'Main/Assets/Project Tiles/Development/Space/VA_dev.png'
        ] ,
    '4' : [
        'Main/Assets/Background/Sea.jpg' , 
        '# Every sea is just a huge Pond' , 
        'If you dont like the Sea theme, you can change it by clicking on the `Generate Images` Button' ,
        'Main/Assets/Project Tiles/ML_DL_AI/Sea/AI Gen-Text Detection.png' ,
        'Main/Assets/Project Tiles/ML_DL_AI/Sea/PII Detection.png' , 
        'Main/Assets/Project Tiles/ML_DL_AI/Sea/HUBMAP.png' ,
        'Main/Assets/Project Tiles/ML_DL_AI/Sea/LLM Exam.png' ,
        'Main/Assets/Project Tiles/Research/Sea/VAl_rind.png' ,
        'Main/Assets/Project Tiles/Development/Sea/VA_dev.png'
        ] ,
    '5' : [
        'Main/Assets/Background/Plane.jpg' , 
        '# No, your phone will not survive that high' ,
        'If you dont like the Plane theme, you can change it by clicking on the `Generate Images` Button' , 
        'Main/Assets/Project Tiles/ML_DL_AI/Plane/AI Gen-Text Detection.png' ,
        'Main/Assets/Project Tiles/ML_DL_AI/Plane/PII Detection.png' , 
        'Main/Assets/Project Tiles/ML_DL_AI/Plane/HUBMAP.png' ,
        'Main/Assets/Project Tiles/ML_DL_AI/Plane/LLM Exam.png' ,
        'Main/Assets/Project Tiles/Research/Plane/VAl_rind.png' ,
        'Main/Assets/Project Tiles/Development/Plane/VA_dev.png'
        ] ,
    '6' : [
        'Main/Assets/Background/Mountain.jpg' , 
        '# Mountain - Where the air is thin and so are the chances of survival' , 
        'If you dont like the Mountain theme, you can change it by clicking on the `Generate Images` Button' ,
        'Main/Assets/Project Tiles/ML_DL_AI/Mountain/AI Gen-Text Detection.png' ,
        'Main/Assets/Project Tiles/ML_DL_AI/Mountain/PII Detection.png' , 
        'Main/Assets/Project Tiles/ML_DL_AI/Mountain/HUBMAP.png' ,
        'Main/Assets/Project Tiles/ML_DL_AI/Mountain/LLM Exam.png' ,
        'Main/Assets/Project Tiles/Research/Mountain/VAl_rind.png' ,
        'Main/Assets/Project Tiles/Development/Mountain/VA_dev.png'
        ] ,
    '7' : [
        'Main/Assets/Background/WaterFall.jpg' , 
        '# WaterFall - Natures way to felx' , 
        'If you dont like the WaterFall theme, you can change it by clicking on the `Generate Images` Button' ,
        'Main/Assets/Project Tiles/ML_DL_AI/WaterFall/AI Gen-Text Detection.png' ,
        'Main/Assets/Project Tiles/ML_DL_AI/WaterFall/PII Detection.png' , 
        'Main/Assets/Project Tiles/ML_DL_AI/WaterFall/HUBMAP.png' ,
        'Main/Assets/Project Tiles/ML_DL_AI/WaterFall/LLM Exam.png' ,
        'Main/Assets/Project Tiles/Research/WaterFall/VAl_rind.png' ,
        'Main/Assets/Project Tiles/Development/WaterFall/VA_dev.png'
        ] ,
    '8' : [
        'Main/Assets/Background/Fish.png' , 
        '# No, they cannot swim in stomach' , 
        'If you dont like the Fish theme, you can change it by clicking on the `Generate Images` Button' ,
        'Main/Assets/Project Tiles/ML_DL_AI/Fish/AI Gen-Text Detection.png' ,
        'Main/Assets/Project Tiles/ML_DL_AI/Fish/PII Detection.png' , 
        'Main/Assets/Project Tiles/ML_DL_AI/Fish/HUBMAP.png' ,
        'Main/Assets/Project Tiles/ML_DL_AI/Fish/LLM Exam.png' ,
        'Main/Assets/Project Tiles/Research/Fish/VAl_rind.png' ,
        'Main/Assets/Project Tiles/Development/Fish/VA_dev.png'
        ] ,
    '9' : [
        'Main/Assets/Background/Desert.jpg' , 
        '# Desert - Where nature hosts Mirage fashion shows' , 
        'If you dont like the Desert theme, you can change it by clicking on the `Generate Images` Button' , 
        'Main/Assets/Project Tiles/ML_DL_AI/Desert/AI Gen-Text Detection.png' ,
        'Main/Assets/Project Tiles/ML_DL_AI/Desert/PII Detection.png' , 
        'Main/Assets/Project Tiles/ML_DL_AI/Desert/HUBMAP.png' ,
        'Main/Assets/Project Tiles/ML_DL_AI/Desert/LLM Exam.png' ,
        'Main/Assets/Project Tiles/Research/Desert/VAl_rind.png' ,
        'Main/Assets/Project Tiles/Development/Desert/VA_dev.png' 
        ]
}

def home() : 
    '''
    Home Page
    '''   

    theme_number = open('Main/Assets/TextFiles/Theme.txt').read()
    set_bg_hack(themes[str(theme_number)][0])

    st.title('Ayush Singhal ')
    st.write('Kaggle Notebooks Master')

    for _ in range(7) : st.markdown('')

    st.markdown(open('Main/Assets/TextFiles/About Me.txt').read() , unsafe_allow_html = True)
    st.markdown(open('Main/Assets/TextFiles/About Me Markdown.txt').read() , unsafe_allow_html=True)

    for _ in range(7) : st.markdown('')

    st.markdown(open('Main/Assets/TextFiles/Skills.txt').read() , unsafe_allow_html = True)
    st.markdown(open('Main/Assets/TextFiles/Skills Markdown.txt').read() , unsafe_allow_html = True)

    classifier = NeuralNetwork()
    classifier.load_state_dict(torch.load('Main/Assets/Models/Classifier Model' , map_location = torch.device('cpu')))
    generator = torch.load('Main/Assets/Models/Generator Model' , map_location = torch.device('cpu'))
    st.write('Stats can take some time to load')

    col_1 , col_2 = st.columns(2)


    if col_1.button('Generate Images') : 

        latent = torch.randn(1 , 128 , 1 , 1)

        fake_image = generator(latent)
        label = classifier(fake_image)

        save_image(fake_image, 'Generated.jpeg')
        col_1.image('Generated.jpeg', width=200, caption=f'Image of {torch.argmax(label)} Generated by GAN')

        with open('Main/Assets/TextFiles/Theme.txt' , 'w') as file : file.write(str(torch.argmax(label).item()))
        theme_number = open('Main/Assets/TextFiles/Theme.txt').read()

        col_2.markdown(themes[str(theme_number)][1])

        for _ in range(16) : col_2.markdown('')

    set_bg_hack(themes[str(theme_number)][0])

    if col_1.button('Sources') : 

        st.write(theme_number)
        for _ in range(7) : col_2.markdown('')

        col_2.code(open('Main/Assets/Code/Classifier.py').read())
        col_1.code(open('Main/Assets/Code/GAN.py').read())
        with col_2 : components.iframe('https://api.wandb.ai/links/ayushsinghal659/i4ta3isp', height=600 , scrolling = True)
        with col_1 : components.iframe('https://api.wandb.ai/links/ayushsinghal659/tm39kito', height=600 , scrolling = True)

    st.markdown(open('Main/Assets/TextFiles/Projects.txt').read() , unsafe_allow_html = True)

    col_1 , col_2  = st.columns(2)

    
    with col_1 : 

        st.write('Use Navigation to go to specific project')
        st.markdown(open('Main/Assets/TextFiles/Deep Learning Projects.txt').read() , unsafe_allow_html = True)
        st.markdown(f'''
            <img 
                src="data:image/jpeg;base64,{get_image_base64_str(themes[str(theme_number)][3])}" 
                width="300" 
                height="200">
        ''' , unsafe_allow_html = True)
        for _ in range(4) : st.markdown('')
        st.markdown(f'''
            <img
                src="data:image/jpeg;base64,{get_image_base64_str(themes[str(theme_number)][4])}"
                width="300"
                height="200">
        ''' , unsafe_allow_html = True)

        st.markdown(open('Main/Assets/TextFiles/Research Projects.txt').read() , unsafe_allow_html = True)
        st.markdown(f'''
            <img
                src="data:image/jpeg;base64,{get_image_base64_str(themes[str(theme_number)][7])}"
                width="300" 
                height="200"> 
        ''' , unsafe_allow_html = True)

        st.markdown(open('Main/Assets/TextFiles/Development Projects.txt').read() , unsafe_allow_html = True)
        st.markdown(f'''
            <img 
                src="data:image/jpeg;base64,{get_image_base64_str(themes[str(theme_number)][8])}" 
                width="300" 
                height="200">
        ''' , unsafe_allow_html = True)

    with col_2 : 

        for _ in range(6) : st.markdown('')
        st.markdown(f'''
            <img
                src="data:image/jpeg;base64,{get_image_base64_str(themes[str(theme_number)][5])}"
                width="300"
                height="200">
        ''' , unsafe_allow_html = True)

        # for _ in range(4) : st.markdown('')
        # st.markdown(f'''
        # <a href = 'https://www.google.com'>
        #     <img
        #         src="data:image/jpeg;base64,{get_image_base64_str(themes[str(theme_number)][6])}"
        #         width="400"
        #         height="200">
        # </a>
        # ''' , unsafe_allow_html = True)


    st.markdown(open('Main/Assets/TextFiles/Accomplishments.txt').read() , unsafe_allow_html = True)
    st.markdown(open('Main/Assets/TextFiles/Accomplishments Markdown.txt').read() , unsafe_allow_html = True)
    st.markdown(open('Main/Assets/TextFiles/How To Reach Me.txt').read() , unsafe_allow_html = True)
def image_to_music() : 
    '''
    Image to Music Page
    '''

    theme_number = open('Main/Assets/TextFiles/Theme.txt').read()
    set_bg_hack(themes[str(theme_number)][0])
    
    st.markdown(open('Main/Assets/TextFiles/AI Gen-Text Detection.txt').read())

    image_in = st.file_uploader(
        'Image reference' , 
        type = ['png' , 'jpg' , 'jpeg'])

    if image_in : 

        image = Image.open(image_in)
        image.save('Main/Assets/Generated/Gen.jpg')

        if st.button('Make music from my pic !') :

            st.write('Generating Captions !')
            caption = get_caption(image_in)

            st.write('Generating Music !')
            path = get_music(caption)

            st.audio(path , format = 'audio/mp4')
def pii_detection() :
    '''
    PII Detection Page
    '''
    theme_number = open('Main/Assets/TextFiles/Theme.txt').read()
    set_bg_hack(themes[str(theme_number)][0])
 
    st.markdown(open('Main/Assets/TextFiles/PII Detection.txt').read())

    text = st.text_area('Enter Text')
    col_1 , col_2 = st.columns(2)

    if col_1.button('Detect') : 

        analyzer = AnalyzerEngine()
        analyzer_results = analyzer.analyze(text=text, language="en")

        st.write({
            analyzer_result.entity_type : text[analyzer_result.start : analyzer_result.end]
            for analyzer_result 
            in analyzer_results 
        })
def hubmap() :
    '''
    HUBMAP Page
    '''

    theme_number = open('Main/Assets/TextFiles/Theme.txt').read()
    set_bg_hack(themes[str(theme_number)][0])
    
    st.markdown(open('Main/Assets/TextFiles/HUBMAP.txt').read())

    st.code(open('Main/Assets/Code/Hubmap.py').read())
    components.iframe('https://wandb.ai/ayushsinghal659/uncategorized/reports/HUBMAP--Vmlldzo2ODEwMjg5?accessToken=tbwoj46kvijblyq3ce0s1vwlr3nk5knn89ctlmhg5etbht40604hyzr2b7muu9sj', height=600 , scrolling = True)

# def llm_exam() :

#     theme_number = open('Main/Assets/TextFiles/Theme.txt').read()
#     set_bg_hack(themes[str(theme_number)][0])

#     st.markdown(open('Main/Assets/TextFiles/LLM Exam.txt').read())

def val_rind() :

    theme_number = open('Main/Assets/TextFiles/Theme.txt').read()
    set_bg_hack(themes[str(theme_number)][0])

    st.markdown(open('Main/Assets/TextFiles/VAl_rind.txt').read()) 
def va_dev() :
    
    theme_number = open('Main/Assets/TextFiles/Theme.txt').read()
    set_bg_hack(themes[str(theme_number)][0])

    st.markdown(open('Main/Assets/TextFiles/VA_dev.txt').read())
    
st.sidebar.title('navigation')

option = st.sidebar.selectbox(
    'Go to' , 
    [
        'Home' , 
        'Echos Of The Canvas' , 
        'Behind The Text' , 
        'Beyond The Visible' ,  
        'VAl_rind' , 
        'VA_dev'
    ])

if option == 'Home' : home()
elif option == 'Echos Of The Canvas' : image_to_music()
elif option == 'Behind The Text' : pii_detection()
elif option == 'Beyond The Visible' : hubmap()
# elif option == 'LLM Exam' : llm_exam()
elif option == 'VAl_rind' : val_rind()
elif option == 'VA_dev' : va_dev()
