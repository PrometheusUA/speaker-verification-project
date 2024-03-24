import numpy as np
import pandas as pd

def partition_speaker_ids(embedding_map: dict, test_speakers: set):
    train_embeddings = []
    train_speaker_ids = []
    
    for file_id, embedding in embedding_map.items():
        speaker_id = file_id.split('/')[0]
        if speaker_id not in test_speakers:
            train_embeddings.append(embedding)
            train_speaker_ids.append(speaker_id)
    train_embeddings = np.array(train_embeddings)

    return train_embeddings, train_speaker_ids

def get_speechbrain_sets(embedding_map: dict, train_speaker_ids: list):
    unique_train_ids = np.unique(train_speaker_ids)
    speaker_to_idx = {speaker_id: idx for idx, speaker_id in enumerate(unique_train_ids)}
    modelset = np.array([speaker_to_idx[speaker_id] for speaker_id in train_speaker_ids])
    segset = np.array([file_id.split('/')[1] for file_id in embedding_map if file_id.split('/')[0] in unique_train_ids])

    return modelset, segset