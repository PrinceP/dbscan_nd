#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include "dbscan.h"
// #include "cnpy.h"
#include "vendor/npy.hpp"
#include <filesystem>


#include <cmath>
#include <algorithm>
using namespace std;


void analyzeDistanceDistribution(const std::vector<std::vector<float>>& points) {
    std::vector<float> distances;
    for (size_t i = 0; i < 1000; ++i) {
        for (size_t j = i + 1; j < 1000; ++j) {
            distances.push_back(euclideanDistance(points[i], points[j]));
        }
    }

    std::sort(distances.begin(), distances.end());
    for (size_t i = 0; i < std::min(distances.size(), size_t(100)); ++i) {
        std::cout << "Distance " << i << ": " << distances[i] << std::endl;
    }
}

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


int main() {
    std::string base_dir = "./python/EEN-RnD-FaceRecognition/face_clustering/Test_data/IMFDB_final_aligned_npy_face_features/";
    std::vector<std::string> labels;
    std::vector<std::vector<float>> data = load_npy_files(base_dir, labels);

    
    // Analyze distance distribution to determine a reasonable eps
    // analyzeDistanceDistribution(data);

    double eps = 1.0;  // Adjust epsilon as needed
    int minPts = 5;  // Adjust minPts as needed

    DBSCAN dbscan(eps, minPts);
    dbscan.fit(data);

    const vector<int>& clusterLabels = dbscan.getLabels();

    // Print clusters
    for (size_t i = 0; i < clusterLabels.size(); ++i) {
        cout << "Embedding " << i << " (Movie star: " << labels[i] << ") is in cluster " << clusterLabels[i] << endl;
    }

    return 0;
}
