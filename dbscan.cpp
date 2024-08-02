#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include "dbscan.h"
#include <nanoflann/nanoflann.hpp>
#include <memory> // Include memory for std::make_unique

using namespace std;
using namespace nanoflann;

double euclideanDistance(const vector<float>& point1, const vector<float>& point2) {
    double sum = 0.0;
    for (size_t i = 0; i < point1.size(); ++i) {
        sum += (point1[i] - point2[i]) * (point1[i] - point2[i]);
    }
    return sum;
}

const int DBSCAN::UNCLASSIFIED = -1;
const int DBSCAN::NOISE = -2;

DBSCAN::DBSCAN(double eps, int minPts) : eps(eps), minPts(minPts), kdTree(nullptr) {}

DBSCAN::~DBSCAN() {
    delete kdTree; // Ensure to free the allocated memory
}

void DBSCAN::fit(const vector<vector<float>>& points) {
    cloud.pts = points;
    kdTree = new KDTree(points[0].size(), cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
    kdTree->buildIndex();

    int n = points.size();
    labels.assign(n, UNCLASSIFIED);

    int clusterId = 0;
    for (int i = 0; i < n; ++i) {
        if (labels[i] == UNCLASSIFIED) {
            if (expandCluster(points, i, clusterId)) {
                ++clusterId;
            }
        }
    }
}

const vector<int>& DBSCAN::getLabels() const {
    return labels;
}

bool DBSCAN::expandCluster(const vector<vector<float>>& points, int pointIdx, int clusterId) {
    vector<int> seeds = regionQuery(points, pointIdx);
    if (seeds.size() < minPts) {
        labels[pointIdx] = NOISE;
        return false;
    }

    for (int seedIdx : seeds) {
        labels[seedIdx] = clusterId;
    }

    seeds.erase(remove(seeds.begin(), seeds.end(), pointIdx), seeds.end());

    while (!seeds.empty()) {
        int currentPointIdx = seeds.back();
        seeds.pop_back();

        vector<int> result = regionQuery(points, currentPointIdx);
        if (result.size() >= minPts) {
            for (int resultIdx : result) {
                if (labels[resultIdx] == UNCLASSIFIED || labels[resultIdx] == NOISE) {
                    if (labels[resultIdx] == UNCLASSIFIED) {
                        seeds.push_back(resultIdx);
                    }
                    labels[resultIdx] = clusterId;
                }
            }
        }
    }

    return true;
}

vector<int> DBSCAN::regionQuery(const vector<vector<float>>& points, int pointIdx) const {
    vector<int> result;
    std::vector<double> queryPt(points[pointIdx].begin(), points[pointIdx].end());
    std::vector<std::pair<size_t, double>> retMatches;
    nanoflann::SearchParams params;

    const size_t nMatches = kdTree->radiusSearch(&queryPt[0], eps * eps, retMatches, params);

    for (size_t i = 0; i < nMatches; ++i) {
        result.push_back(retMatches[i].first);
    }
    return result;
}
