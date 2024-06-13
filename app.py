import streamlit as st

website_url = "https://vamsi-genai.streamlit.app/"

# Define the URLs of the images and the URLs of the pages you want to navigate to
image_urls = [
    ("https://www.ringcentral.com/gb/en/blog/wp-content/uploads/2021/01/artificial-intelligenceai-chat-bot-concept-1536x1024.jpg", website_url+"/Chat_Bot","ChatBot"),
    ("https://tse2.mm.bing.net/th?id=OIP.MgRQ7QtM_TvVtnuwOnD-jAHaEK&pid=Api&P=0&h=220", website_url+"/Content_Generator","Content Generator"),
    ("https://analyticsindiamag.com/wp-content/uploads/2020/05/chatbot_adoption.jpg", website_url+"/Image_to_Text_Description","Image to Text"),
    ("https://aviancetechnologies.com/wp-content/uploads/2022/05/free-meta-tag-generator.jpg", website_url+"/Seo_MetaData_Generator","Seo MetaData Generator"),
    ("https://murf.ai/resources/media/posts/97/concept-program-smartphone-translate-from-different-languages.jpg", website_url+"/Translator","Translator")
]

for image_url, page_url , name in image_urls[:5]:
    st.title(name+":")
    st.markdown(f"<a href='{page_url}' target='_blank'><img src='{image_url}' width='700' height='450'></a>", unsafe_allow_html=True)
      

