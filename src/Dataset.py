import pickle
import os
from collections import Counter
import matplotlib.pyplot as plt

def get_key_counter(dataset_path, save_path='models/key_counter.pkl'):
    all_keys = []

    song_folders = os.listdir(dataset_path)
    songs_processed = -1

    print(f"Total songs in folder: {len(song_folders)}")

    for song_folder in song_folders:
        songs_processed += 1
        song_path = os.path.join(dataset_path, song_folder)

        key_path = os.path.join(song_path, "key_audio.txt")

        if not os.path.exists(key_path):
            continue

        print(f"processing file name: {song_path}. Songs processed: {songs_processed}/{len(song_folders)}")

        try:
            with open(key_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        key = parts[2]
                        all_keys.append(key)
        except Exception as e:
            print(f"Error reading {song_folder}: {e}")
            continue

    counter = Counter(all_keys)
    print(f"total songs processed: {songs_processed}")
    
    # Save the counter to a pickle file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(counter, f)
    print(f"Key counter saved to {save_path}")
    
    return counter

def get_all_unique_chords(dataset_path, save_path='models/unique_chords.pkl'):
    all_chords = []

    song_folders = os.listdir(dataset_path)
    songs_processed = -1

    print(f"Total songs in folder: {len(song_folders)}")

    for song_folder in song_folders:
        songs_processed += 1
        song_path = os.path.join(dataset_path, song_folder)

        chord_path = os.path.join(song_path, "chord_midi.txt")

        if not os.path.exists(chord_path):
            continue

        print(f"processing file name: {song_path}. Songs processed: {songs_processed}/{len(song_folders)}")

        try:
            with open(chord_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        chord_name = parts[2]
                        all_chords.append(chord_name)
        except Exception as e:
            print(f"Error reading {song_folder}: {e}")
            continue

    counter = Counter(all_chords)
    print(f"total songs processed: {songs_processed}")
    
    # Save the counter to a pickle file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(counter, f)
    print(f"Chord counter saved to {save_path}")
    
    return counter

def load_data(file_path):
    with open(file_path, 'rb') as f:
        counter = pickle.load(f)
    return counter

def pie_chart(data):

    labels = []
    sizes = []
    
    for key, count in data.items():
        labels.append(key)
        sizes.append(count)

    plt.pie(sizes,labels=labels)

    plt.axis('equal')
    plt.show()


if __name__ == "__main__":
    keys = load_data('models/key_counter.pkl')
    pie_chart(keys)