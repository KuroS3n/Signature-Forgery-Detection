import numpy as np
from dtw import dtw
from sklearn.metrics import accuracy_score

signature1 = np.array([[1, 2], [2, 3], [3, 4], [5, 6]]) 
signature2 = np.array([[1, 2], [2, 3.5], [3.5, 4], [5, 6.5]])

def compute_dtw_distance(sig1, sig2):
    manhattan_distance = lambda x, y: np.abs(x - y).sum()
    alignment = dtw(sig1, sig2, dist=manhattan_distance)
    return alignment.distance

distance = compute_dtw_distance(signature1, signature2)
print(f"DTW Distance: {distance}")

threshold = 5.0
is_forgery = distance > threshold
print("Forgery!" if is_forgery else "Genuine")