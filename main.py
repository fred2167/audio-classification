import streamlit as st
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import helper
import plot

plt.rcParams['figure.figsize'] = [16.0, 4.8]


def logistic_regression(X, Y):
    model = LogisticRegression(penalty="none") 
    model.fit(X, Y)
    return model

def main():
    folder_path = os.path.join(os.getcwd(), "audio")
    music_paths, speech_paths = helper.getPaths(folder_path)
    plot.displayInfo(speech_paths[1])

if __name__ == "__main__":
    main()