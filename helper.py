import torch
import os
import glob

def getPathkey(path):
  tokens = path.split("/")
  fileName = tokens[-1]
  start =  2
  end = fileName.find(".wav")
  return int(fileName[start:end])

def getPaths(folder_path):
  music_paths = sorted(glob.glob(os.path.join(folder_path, "music/*")), key=getPathkey)
  speech_paths = sorted(glob.glob(os.path.join(folder_path, "speech/*")), key=getPathkey)
  return music_paths, speech_paths

def getFeatures(folder_path):
    music_feature_path = os.path.join(folder_path, "music_feat.pt")
    speech_feature_path = os.path.join(folder_path, "speech_feat.pt")
    return torch.load(music_feature_path), torch.load(speech_feature_path)

