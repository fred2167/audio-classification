import streamlit as st
import os
import helper
import plot



def main_page(folder_path, audio_str):

    music_paths, speech_paths = helper.getPaths(folder_path)

    if audio_str == "Music":
        paths = music_paths
    else:
        paths = speech_paths
    index = st.number_input("Data Index", min_value=0, max_value=19)
    plot.displayInfo(paths[index])
    return index

@st.cache
def ml_prediction(data, model, audio_str, idx):

    if audio_str == "Music":
        query_feat = data["music_feature"][idx]
    else:
        query_feat = data["speech_feature"][idx]

    pred = model.predict(query_feat.clone().view(1, -1))
    pred = pred.astype(int).item()
    return data["idxToLabel"][pred], query_feat

@st.cache
def ml_transcript(decoder, query_feature):
    transcript = decoder(query_feature)
    words = transcript.split("|")
    return " ".join(words)


def main():
    folder_path = os.path.join(os.getcwd(), "audio")
    
    audio_str = st.sidebar.radio("Audio", ["Music", "Speech"])

    query_idx = main_page(folder_path, audio_str)

    data, predictor, decoder = helper.get_data_predictor_decoder(folder_path)
    
    pred_str, query_feature = ml_prediction(data, predictor, audio_str, query_idx) 

    st.sidebar.write("ML Prediction: ", pred_str)

    if audio_str == "Speech":
        transcript = ml_transcript(decoder, query_feature)
        st.write("ML Transcript: ", transcript)


if __name__ == "__main__":
    main()