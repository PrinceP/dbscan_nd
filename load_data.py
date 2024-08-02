import os
import numpy as np

def compute_facepair_dist(embeddings_1, embeddings_2):
    '''Compute face-to-face distance'''
    diff = np.subtract(embeddings_1, embeddings_2)
    dist = np.sqrt(np.sum(np.square(diff), -1)).squeeze()
    return dist

def load_npy_files(base_dir):
    data = []
    labels = []

    for moviestar in os.listdir(base_dir):
        # print(moviestar)
        moviestar_path = os.path.join(base_dir, moviestar)
        if os.path.isdir(moviestar_path):
            for moviename in os.listdir(moviestar_path):
                # print(moviename)
                moviename_path = os.path.join(moviestar_path, moviename)
                # print(moviename_path)
                if os.path.isdir(moviename_path):
                    for npy_file in os.listdir(moviename_path):
                        if npy_file.endswith('.npy'):
                            # print(npy_file)
                            npy_path = os.path.join(moviename_path, npy_file)
                            embedding = np.load(npy_path)
                            data.append(embedding)
                            labels.append(moviestar)

    return np.array(data), labels

def analyzeDistanceDistribution(data, num_samples=100):
    num_points = min(1000, data.shape[0])
    distances = []
    
    for i in range(num_points):
        for j in range(i + 1, num_points):
            dist = compute_facepair_dist(data[i], data[j])
            distances.append(dist)
    
    distances.sort()
    sample_distances = distances[:num_samples]
    
    for idx, dist in enumerate(sample_distances):
        print(f"Distance {idx}: {dist}")


base_dir = './python/EEN-RnD-FaceRecognition/face_clustering/Test_data/IMFDB_final_aligned_npy_face_features/'
data, labels = load_npy_files(base_dir)

analyzeDistanceDistribution(data)


# print(f"Loaded {len(data)} embeddings")
# np.save('embeddings.npy', data)
# with open('labels.txt', 'w') as f:
#     for label in labels:
#         f.write(f"{label}\n")