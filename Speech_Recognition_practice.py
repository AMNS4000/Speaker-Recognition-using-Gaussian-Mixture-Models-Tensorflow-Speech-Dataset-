import librosa
import numpy as np
import noisereduce as nr
import pandas as pd
import os
from sklearn.mixture import GaussianMixture


class MultiGMMFuser:
    def __init__(self,k,bg_noise,class_mappings):
        self.k = k
        self.Gmm_list = {}
        self.labels = {}
        self.bg_noise = bg_noise
        self.combined_features = {}
        self.class_mappings = class_mappings
        self.vad_threshold = 0.02
        self.class_gmms = {}
    
    def preprocess_audio(self,audio_data,sample_rate = 16000):
        filtered_audio = nr.reduce_noise(y = audio_data,sr = sample_rate, stationary = True)
        preemp_audio = librosa.effects.preemphasis(y = filtered_audio)
        rms = librosa.feature.rms(y=preemp_audio).ravel()
        vad_audio = preemp_audio[rms > self.vad_threshold]
        return vad_audio
    
    def extract_features(self,data_path):
        mfcc_features = []
        delta_features = []
        delta_delta_features = []
        combined_features = []
        combined_list_features = []
        labels_list = []
        for class_name,labels in enumerate(self.class_mappings):
            class_directory = data_path + str(class_name)
            mfcc_list = []
            delta_list = []
            delta_delta_list = []
            labels = []
            for filename in os.listdir(class_directory):
                file_path = class_directory + str(filename)
                audio_file, sample_rate = librosa.load(file_path)
                audio_preprocessed = self.preprocess_audio(audio_file,sample_rate)
                frame_len = int(sample_rate*0.025)
                hop_length = int(sample_rate*0.010)
                mfccs = librosa.feature.mfccs(y=audio_preprocessed,sr=sample_rate,hop_length=hop_length,n_mfccs=13,n_fft = frame_len)
                deltas = librosa.feature.delta(mfccs)
                deltas_deltas = librosa.feature.delta(mfccs,order=2)
                # (101,13) --> shape of MFCCS
                mfcc_list.append(mfccs.T)
                delta_list.append(deltas.T)
                delta_delta_list.append(deltas_deltas.T)
                labels.append(class_name)

            mfcc_features = np.concatenate(mfcc_list)
            delta_features = np.concetenate(delta_list)
            delta_delta_features = np.concatenate(delta_delta_list)
            combined_features = np.hstack((mfcc_features,delta_features,delta_delta_features))
            combined_list_features.append(combined_features)
            labels_list.append(labels)

        return combined_list_features,labels_list

    def extract_features_test(self,data_path):
        combined_features = []
        class_directory = data_path
        combined_list_features = []
        for filename in os.listdir(class_directory):
            file_path = class_directory + str(filename)
            audio_file, sample_rate = librosa.load(file_path)
            audio_preprocessed = self.preprocess_audio(audio_file,sample_rate)
            frame_len = int(sample_rate*0.025)
            hop_length = int(sample_rate*0.010)
            mfccs = librosa.feature.mfccs(y=audio_preprocessed,sr=sample_rate,hop_length=hop_length,n_mfccs=13,n_fft = frame_len)
            deltas = librosa.feature.delta(mfccs)
            deltas_deltas = librosa.feature.delta(mfccs,order=2)
            # (101,13) --> shape of MFCCS
            combined_features = np.hstack((mfccs.T,deltas.T,deltas_deltas.T))
            combined_list_features.append(combined_features)

        return combined_list_features      

    def fit(self,data_path):
        combined_features,labels_list = self.extract_features(data_path)
        class_gmms = {}
        for class_label,features in zip(combined_features,labels_list):
            gmm = GaussianMixture(n_components=1,covariance_type='full',random_state=42)
            gmm.fit(features)
            class_gmms[labels_list[0]] = gmm
        self.class_gmms = class_gmms

    def predict(self,data_path):
        combined_features = self.extract_features_test(data_path)
        predicted_labels= []
        for features in combined_features:
            likelihoods = {}
            for class_labels,gmm_model in self.class_gmms.items():
                likelihood = gmm_model.score(features)
                likelihoods[class_labels] = likelihood
            predicted_labels.append(max(likelihoods,key=likelihoods.get))
        return predicted_labels
    
    





