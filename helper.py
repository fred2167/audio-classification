import torch
import os
import glob

def _getPathkey(path):
  tokens = path.split("/")
  fileName = tokens[-1]
  start =  2
  end = fileName.find(".wav")
  return int(fileName[start:end])

def getPaths(folder_path):
  music_paths = sorted(glob.glob(os.path.join(folder_path, "music/*")), key=_getPathkey)
  speech_paths = sorted(glob.glob(os.path.join(folder_path, "speech/*")), key=_getPathkey)
  return music_paths, speech_paths

def getFeatures(folder_path):
    music_feature_path = os.path.join(folder_path, "music_feat.pt")
    speech_feature_path = os.path.join(folder_path, "speech_feat.pt")
    return torch.load(music_feature_path), torch.load(speech_feature_path)

def getDataset(folder_path, num_train_examples = 15):
    torch.seed(1234)

    music_feature, speech_feature = getFeatures(folder_path)
    music_labels = torch.zeros((num_train_examples, 1))
    speech_labels= torch.ones((num_train_examples, 1))
    idxToLabel = ["Music", "Speech"]

    x_train = torch.concat((music_feature[:num_train_examples,:], speech_feature[:num_train_examples,:]))
    y_train = torch.concat((music_labels, speech_labels))

    idx = torch.randperm(num_train_examples) 
    x_train = x_train[idx, :]
    y_train = y_train[idx].squeeze()

    num_test_examples = music_feature.shape[0] + speech_feature.shape[0] - 2*num_test_examples
    x_test = torch.concat((music_feature[num_train_examples:, :], speech_feature[num_train_examples:, :])) 
    y_test = torch.concat((music_labels[:num_test_examples], speech_labels[:num_test_examples]))

    data = {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test, "idxToLabel": idxToLabel}
    return data





