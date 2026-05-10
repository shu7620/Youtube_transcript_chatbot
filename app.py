# import streamlit as st
# from logic import process_video_to_vectorstore,query_vectorstores

# st.set_page_config(page_title="Pro YT Chatbot", page_icon="🔍")
# st.title("🔍 Semantic Video Search (FAISS)")

# if 'vectorstore' not in st.session_state:
#     st.session_state.vectorstore=None
# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history=[]

# with st.sidebar:
#     st.header("Settings")
#     url=st.text_input("YouTuve video URL")

#     if st.button("Process Video"):
#         if url:
#             with st.spinner("Processing video..."):
#                 try:
#                     # Logic call :Create FAISS index
#                     st.session_state.vectorstore=process_video_to_vectorstore(url)
#                     st.session_state.chat_history=[]
#                     st.success("Video Indexed to FAISS!")
#                 except Exception as e:
#                     st.error(f"Error processing video: {str(e)}")


# # Main Chat Interface
# for mssg in st.session_state.chat_history:
#     with st.chat_message(mssg['role']):
#         st.markdown(mssg['content'])

# if prompt :=st.chat_input("Ask about the video..."):
#     if not st.session_state.vectorstore:
#         st.error("Please process a video first!")
#     else:
#         st.session_state.chat_history.append({'role': 'user','content':prompt})
#         with st.chat_message('user'):
#             st.markdown(prompt)

#         with st.chat_message("assistant"):
#             with st.spinner("Searching and Generating..."):
#                 answer=query_vectorstores(st.session_state.vectorstore,prompt)
#                 st.markdown(answer)
#                 st.session_state.chat_history.append({'role':'assistant','content':answer})

import streamlit as st

from logic import process_video_to_vectorstore, query_vectorstores

st.set_page_config(page_title="Pro YT Chatbot", page_icon="🔍")
st.title("🔍 Semantic Video Search (FAISS)")

# Ensure session state is initialized
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.header("Settings")
    

    url = st.text_input("YouTube video URL") # Fixed "YouTuve" typo

    if st.button("Process Video"):
        if url:
            with st.spinner("Processing video..."):
                try:
                    # Pass both url AND api_key to match logic.py
                    st.session_state.vectorstore = process_video_to_vectorstore(url)
                    st.session_state.chat_history = []
                    st.success("Video Indexed to FAISS!")
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")
        else:
            st.warning("Please provide both the API Key and URL.")

# Main Chat Interface
for mssg in st.session_state.chat_history:
    with st.chat_message(mssg['role']):
        st.markdown(mssg['content'])

if prompt := st.chat_input("Ask about the video..."):
    # FIX: Use 'is None' check to prevent the "process a video first" false-positive
    if st.session_state.vectorstore is None:
        st.error("Please process a video first!")
    else:
        st.session_state.chat_history.append({'role': 'user', 'content': prompt})
        with st.chat_message('user'):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching and Generating..."):
                try:
                    # Pass api_key here as well
                    answer = query_vectorstores(st.session_state.vectorstore, prompt)
                    st.markdown(answer)
                    st.session_state.chat_history.append({'role': 'assistant', 'content': answer})
                except Exception as e:
                    st.error(f"Chat Error: {e}")