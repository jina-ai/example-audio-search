import streamlit as st
import tempfile
import os
import requests
from math import ceil,floor

DEFAULT_ENDPOINT = 'http://127.0.0.1:45678/search'

APP_NAME = 'Audio Search'
APP_DESCRIPTION = 'Looking For Audio Through Text & Audio'

TOP_K = 10

# Query For Data
def request_and_display(query_input, top_k):
    print(query_input)
    headers = {
        'Content-Type': 'application/json',
    }

    doc = {}
    search_type = 0
    if query_input.endswith('.mp3') and os.path.exists(query_input):
        doc['uri'] = query_input
        # doc['mime_type'] = 'image/jpeg'
    else:
        doc['text'] = str(query_input)
        doc['mime_type'] = 'text/plain'
        search_type = 1

    body = {
        'data': [doc],
        # 'parameters': {
        #     'limit': top_k
        # }
    }
    response = requests.post(DEFAULT_ENDPOINT, headers=headers, json=body)
    content = response.json()
    matches = content['data']['docs'][0]['matches']
    print(len(matches))
    display_audios(matches, search_type)


# Display match photos:
def display_audios(matches, search_type):
    st.markdown(
        f"""
        ### Best matches for your query:
        """
    )
    cnt = len(matches)
    row = int(ceil(cnt / 2))
    for i in range(row):
        col1, col2 = st.columns(2)
        with col1:
            if i * 2 < cnt:
                # search_type 1:search with text  0:search with image
                if search_type:
                    uri = matches[i * 2]['uri']
                else:
                    uri = matches[i * 2]['uri']
                begin = floor(matches[i * 2]['tags']['beg_in_ms']/1000)
                score = matches[i * 2]["scores"]['cosine']['value']
                st.markdown(f"#### No.{i * 2 + 1}: Score  {score:.6f}\n")
                st.audio(uri, start_time = begin)
        with col2:
            if i * 2 + 1 < cnt:
                # search_type 1:search with text  0:search with image
                if search_type:
                    uri = matches[i * 2 + 1]['uri']
                else:
                    uri = matches[i * 2 + 1]['uri']
                begin = floor(matches[i * 2 + 1]['tags']['beg_in_ms']/1000)
                score = matches[i * 2 + 1]['scores']['cosine']['value']
                st.markdown(f"####  No.{i * 2 + 2}:Score  {score:.6f}\n")
                st.audio(uri, start_time = begin)


def main():
    # Header
    st.markdown(
        f"""
        # <a href="https://github.com/jina-ai/jina/">Jina</a>\'s {APP_NAME}
        ### {APP_DESCRIPTION}
        """,
        unsafe_allow_html=True,
    )

    # Sidebar
    st.sidebar.header('Using audio or text to search audio.')
    input_choice = st.sidebar.selectbox(
        'Please choose the type of your input:',
        [{'name': 'search_by_audio', 'display': 'upload a piece of audio and search for similar files'},
         {'name': 'search_by_text', 'display': 'input text and search for audio'}],
        format_func=lambda x: x['display']
    )['name']
    settings = st.sidebar.expander(label='Settings', expanded=False)
    with settings:
        endpoint = st.text_input(label='Endpoint', value=DEFAULT_ENDPOINT)
        top_k = st.number_input(label='Top K', value=TOP_K, step=1)
    st.sidebar.button('select')

    st.sidebar.markdown(
        f"""
        **This is a {APP_NAME} using the [Jina neural search framework](https://github.com/jina-ai/jina/).**
        You can search for audios using text or similar voice pieces!

        <a href="https://github.com/jina-ai/jina/"><img src="https://github.com/alexcg1/jina-app-store-example/blob/a8f64332c6a5b3ae42df07d4bd615ff1b7ece4d9/frontend/powered_by_jina.png?raw=true" width=256></a>
        """,
        unsafe_allow_html=True,
    )

    query_text = None
    query_audio = None
    query_audio_path = None
    # Content
    if input_choice == 'search_by_text':
        query_text = st.text_input(
            label='please input your search content'
        )
    elif input_choice == 'search_by_audio':
        query_audio = st.file_uploader(
            label='upload one audio file',
        )
        if query_audio is not None:
            query_audio_path = './toy-data/'+query_audio.name
            st.audio(query_audio)

    clicked = st.button("Search")
    if clicked:
        if (not query_text) and (not query_audio):
            st.markdown('Please enter a query')
        else:
            request_and_display(query_audio_path or query_text, top_k)
            # if query_audio_file:
            #     query_audio_file.close()


if __name__ == '__main__':
    main()
