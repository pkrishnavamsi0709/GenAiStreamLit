import google.generativeai as genai
import streamlit as st

genai.configure(api_key="AIzaSyDcF1LrSLzb9l3B7NfS_5LFNyoGnMv6K_g")


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

def askgenaibot(Query):
    # data = json.loads(request.data)
    usertext = Query
    model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
                              generation_config=generation_config,
                              safety_settings=safety_settings)
    print("UserQuery:",usertext)
    response = model.generate_content(f"Generate Content {usertext}")
    return response.text

st.title('Ask GPT-3 AI Bot')
st.sidebar.success("select a page above")

usertext = st.text_input("Enter your query:")

if st.button("Generate Response"):
    model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
                                  generation_config=generation_config,
                                  safety_settings=safety_settings)
    response = model.generate_content(f"Generate Content {usertext}")
    st.write(response.text)
    
