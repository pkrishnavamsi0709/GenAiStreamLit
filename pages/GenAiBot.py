import google.generativeai as genai
import os
import streamlit as st
from dotenv import load_dotenv
from langchain import PromptTemplate
import speech_recognition as sr
import pyaudio
 
model = genai.GenerativeModel('gemini-pro')
genai.configure(api_key="AIzaSyDcF1LrSLzb9l3B7NfS_5LFNyoGnMv6K_g")
 
st.set_page_config(page_title="Chatbot", page_icon="🤖")
 
st.header("Chatbot 🤖")
 
generation_config = {
  "temperature": 0.9,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 2048,
}
 
safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
]
 
prompt_template: str = """/
Use the following pieces of context to answer the question/
question: {question}.
say Thank you....! at the end/
"""
 
prompt = PromptTemplate.from_template(template=prompt_template)
 
def askgenaibot(input):
    model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
                              generation_config=generation_config,
                              safety_settings=safety_settings)
    response = model.generate_content(f"Generate Content {input}")
    return response.text
 
def speech_to_text():
    recognizer = sr.Recognizer()
 
    with sr.Microphone() as source:
        st.write("Speak something...")
        # Adjust for ambient noise
        recognizer.adjust_for_ambient_noise(source)
        # Listen for audio input
        audio = recognizer.listen(source)
    try:
        # Use Google Speech Recognition
        text = recognizer.recognize_google(audio)
        st.write("Your input:", text)
        return text
    except sr.UnknownValueError:
        st.write("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        st.write("Could not request results from Google Speech Recognition service; {0}".format(e))
speak=""
with st.sidebar:
    st.header("Chat through voice 🎤")
    if st.button("speak"):        
        text = speech_to_text()
        speak = text
        if text:
                response = model.generate_content(text)
                st.write(f"Translation: {response.text}")
        else:
            st.write("Please enter text to Genrate.")
 
if speak is not None:
    input = st.text_input("Input Prompt: ",key="input")
    prompt_formatted_str: str = prompt.format(question=input)
    if st.button("search"):
        response=askgenaibot(prompt_formatted_str)
        st.subheader("The Response is")
        st.write(response)
 
 
has context menu
