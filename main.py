import streamlit as st
import os
import matplotlib.pyplot as plt
import helper
import plot



def main():
    plt.rcParams['figure.figsize'] = [16.0, 4.8]
    folder_path = os.path.join(os.getcwd(), "audio")
    music_paths, speech_paths = helper.getPaths(folder_path)
    plot.desplayInfo(music_paths[0])

if __name__ == "__main__":
    main()