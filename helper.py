import streamlit as st
import torch
import torchaudio
import os
import glob
from sklearn.linear_model import LogisticRegression

def _getPathkey(path):
  tokens = path.split("/")
  fileName = tokens[-1]
  start =  2
  end = fileName.find(".wav")
  return int(fileName[start:end])

@st.cache
def getPaths(folder_path):
  music_paths = sorted(glob.glob(os.path.join(folder_path, "music/*")), key=_getPathkey)
  speech_paths = sorted(glob.glob(os.path.join(folder_path, "speech/*")), key=_getPathkey)
  return music_paths, speech_paths

@st.cache
def pathsToTensors(paths, normalize=True):

  out = []
  for path in paths:
    waveform,_ = torchaudio.load(path, normalize=normalize)
    out.append(waveform[:,:60000].squeeze())
  
  return torch.stack(out)

def getEncoderDecoder():
    class Decoder(torch.nn.Module):
        def __init__(self, labels, ignore):
            super().__init__()
            self.labels = labels
            self.ignore = ignore

        def forward(self, emission: torch.Tensor) -> str:
            """Given a sequence emission over labels, get the best path string
            Args:
            emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

            Returns:
            str: The resulting transcript
            """
            indices = torch.argmax(emission, dim=-1)  # [num_seq,]
            indices = torch.unique_consecutive(indices, dim=-1)
            indices = [i for i in indices if i not in self.ignore]
            return ''.join([self.labels[i] for i in indices])

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    encoder = bundle.get_model()
    decoder = Decoder(bundle.get_labels(), ignore=(0, 1, 2, 3)) 
    return encoder, decoder

@st.cache
def getFeatures(folder_path):
    music_feature_path = os.path.join(folder_path, "music_feat.pt")
    speech_feature_path = os.path.join(folder_path, "speech_feat.pt")
    if os.path.exists(music_feature_path) and os.path.exists(speech_feature_path):
        return torch.load(music_feature_path), torch.load(speech_feature_path)
    
    music_paths, speech_paths = getPaths(folder_path)
    music_tensors = pathsToTensors(music_paths)
    speech_tensors = pathsToTensors(speech_paths)

    encoder, _ = getEncoderDecoder()
    with torch.inference_mode():
        music_features, _ = encoder(music_tensors)
        speech_features, _ = encoder(speech_tensors)
    
    return music_features, speech_features

@st.cache
def getDataset(folder_path, num_train_examples = 15):

    music_feature, speech_feature = getFeatures(folder_path)
    music_labels = torch.zeros((num_train_examples, 1))
    speech_labels= torch.ones((num_train_examples, 1))
    idxToLabel = ["Music", "Speech"]

    x_train = torch.concat((music_feature[:num_train_examples,:], speech_feature[:num_train_examples,:]))
    y_train = torch.concat((music_labels, speech_labels)).squeeze()

    num_test_examples = music_feature.shape[0] + speech_feature.shape[0] - 2*num_train_examples
    x_test = torch.concat((music_feature[num_train_examples:, :], speech_feature[num_train_examples:, :])) 
    y_test = torch.concat((music_labels[:num_test_examples], speech_labels[:num_test_examples]))


    data = {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test, "idxToLabel": idxToLabel, \
            "music_feature": music_feature, "speech_feature": speech_feature}
    return data



@st.cache
def get_data_predictor_decoder(folder_path):
    data = getDataset(folder_path)

    X, Y = data["x_train"], data["y_train"]

    model = LogisticRegression(penalty="none") 
    if len(X.shape) > 2:
        X = X.reshape(X.shape[0], -1)
    if len(Y.shape) > 2:
        Y = Y.reshape(Y.shape[0], -1)
    model.fit(X, Y)

    _, decoder = getEncoderDecoder()
    return data, model, decoder 

