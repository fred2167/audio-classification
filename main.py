import streamlit as st
import os
from sklearn.linear_model import LogisticRegression
import helper
import plot


@st.cache
def logistic_regression(X, Y):
    model = LogisticRegression(penalty="none") 
    if len(X.shape) > 2:
        X = X.reshape(X.shape[0], -1)
    if len(Y.shape) > 2:
        Y = Y.reshape(Y.shape[0], -1)
    model.fit(X, Y)
    return model

def main():
    folder_path = os.path.join(os.getcwd(), "audio")
    music_paths, speech_paths = helper.getPaths(folder_path)

    audio_str = st.sidebar.radio("Audio", ["Music", "Speech"])
    if audio_str == "Music":
        paths = music_paths
    else:
        paths = speech_paths
    index = st.number_input("Data Index", min_value=0, max_value=19)
    plot.displayInfo(paths[index])

    # data = helper.getDataset(folder_path)
    # model = logistic_regression(data["x_train"], data["y_train"])
    # print(data["y_train"])

if __name__ == "__main__":
    main()