import torch
import torchaudio
import matplotlib.pyplot as plt
import librosa
import streamlit as st
import seaborn as sns
import helper

# sns.set(font_scale=2)
plt.rcParams['figure.figsize'] = [16.0, 4.8]

def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  time_axis = torch.arange(0, num_frames) / sample_rate

  figure, axes = plt.subplots(num_channels, 1)
  if num_channels == 1:
    axes = [axes]
  for c in range(num_channels):
    axes[c].plot(time_axis, waveform[c], linewidth=1)
    axes[c].grid(True)
    if num_channels > 1:
      axes[c].set_ylabel(f'Channel {c+1}')
    if xlim:
      axes[c].set_xlim(xlim)
    if ylim:
      axes[c].set_ylim(ylim)
  figure.suptitle(title)
  st.pyplot(figure)

def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  time_axis = torch.arange(0, num_frames) / sample_rate

  figure, axes = plt.subplots(num_channels, 1)
  if num_channels == 1:
    axes = [axes]
  for c in range(num_channels):
    axes[c].specgram(waveform[c], Fs=sample_rate)
    if num_channels > 1:
      axes[c].set_ylabel(f'Channel {c+1}')
    if xlim:
      axes[c].set_xlim(xlim)
  figure.suptitle(title)
  st.pyplot(figure)

def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
  fig, axs = plt.subplots(1, 1)
  axs.set_title(title or 'Spectrogram (db)')
  axs.set_ylabel(ylabel)
  axs.set_xlabel('frame')
  im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
  if xmax:
    axs.set_xlim((0, xmax))
  fig.colorbar(im, ax=axs)
  st.pyplot(fig)

def play_audio(path):
    st.audio(path)

def plot_confusion_matrix(data, model):
    confusion_matrix = helper.getConfusionMatrix(data, model)
    fig = plt.Figure(figsize= (10, 10))
    ax = fig.add_subplot()
    kw = {"ax": ax, "annot": True, "cbar": False, "annot_kws": {"fontsize":30}, "cmap":"GnBu"}
    sns.heatmap(confusion_matrix, **kw)
    ax.set_xticklabels(data["idxToLabel"], fontsize=30)
    ax.set_yticklabels(data["idxToLabel"], fontsize=30)
    st.sidebar.write("Confusion Matrix: ")
    st.sidebar.pyplot(fig)

def displayInfo(path):
  waveform, sample_rate = torchaudio.load(path)
  plot_waveform(waveform, sample_rate)
  plot_specgram(waveform, sample_rate)
  play_audio(path)
