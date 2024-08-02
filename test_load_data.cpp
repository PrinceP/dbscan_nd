#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include "vendor/npy.hpp"  // Include the libnpy header

namespace fs = std::filesystem;

std::vector<std::vector<float>> load_npy_files(const std::string& base_dir, std::vector<std::string>& labels) {
    std::vector<std::vector<float>> data;

    for (const auto& moviestar_entry : fs::directory_iterator(base_dir)) {
        if (moviestar_entry.is_directory()) {
            std::string moviestar = moviestar_entry.path().filename().string();
            for (const auto& moviename_entry : fs::directory_iterator(moviestar_entry.path())) {
                if (moviename_entry.is_directory()) {
                    for (const auto& npy_file_entry : fs::directory_iterator(moviename_entry.path())) {
                        if (npy_file_entry.path().extension() == ".npy") {
                            std::string npy_path = npy_file_entry.path().string();
                            npy::npy_data d = npy::read_npy<float>(npy_path);  // Correct API call
                            std::vector<float> embedding = d.data;
                            data.push_back(embedding);
                            labels.push_back(moviestar);
                        }
                    }
                }
            }
        }
    }

    return data;
}

double compute_facepair_dist(const std::vector<float>& embeddings_1, const std::vector<float>& embeddings_2) {
    double sum = 0.0;
    for (size_t i = 0; i < embeddings_1.size(); ++i) {
        double diff = embeddings_1[i] - embeddings_2[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

void analyzeDistanceDistribution(const std::vector<std::vector<float>>& data, size_t num_samples = 100) {
    size_t num_points = std::min<size_t>(1000, data.size());
    std::vector<double> distances;

    for (size_t i = 0; i < num_points; ++i) {
        for (size_t j = i + 1; j < num_points; ++j) {
            double dist = compute_facepair_dist(data[i], data[j]);
            distances.push_back(dist);
        }
    }

    std::sort(distances.begin(), distances.end());
    size_t sample_size = std::min(num_samples, distances.size());

    for (size_t idx = 0; idx < sample_size; ++idx) {
        std::cout << "Distance " << idx << ": " << distances[idx] << std::endl;
    }
}

int main() {
    std::string base_dir = "./python/EEN-RnD-FaceRecognition/face_clustering/Test_data/IMFDB_final_aligned_npy_face_features/";
    std::vector<std::string> labels;
    std::vector<std::vector<float>> data = load_npy_files(base_dir, labels);

    // Debug: Print a sample of the data to verify correct loading
    std::cout << "Loaded " << data.size() << " embeddings." << std::endl;
    for (size_t i = 0; i < std::min<size_t>(5, data.size()); ++i) {
        std::cout << "Embedding " << i << ": ";
        for (size_t j = 0; j < std::min<size_t>(5, data[i].size()); ++j) {
            std::cout << data[i][j] << " ";
        }
        std::cout << std::endl;
    }

    analyzeDistanceDistribution(data);

    return 0;
}