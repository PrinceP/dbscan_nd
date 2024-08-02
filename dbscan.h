#ifndef DBSCAN_H
#define DBSCAN_H

#include <vector>
#include <nanoflann/nanoflann.hpp>
#include <memory> // Include memory for std::unique_ptr

double euclideanDistance(const std::vector<float>& point1, const std::vector<float>& point2);

class DBSCAN {
public:
    DBSCAN(double eps, int minPts);
    ~DBSCAN(); // Add destructor
    void fit(const std::vector<std::vector<float>>& points);
    const std::vector<int>& getLabels() const;

private:
    static const int UNCLASSIFIED;
    static const int NOISE;

    double eps;
    int minPts;
    std::vector<int> labels;

    bool expandCluster(const std::vector<std::vector<float>>& points, int pointIdx, int clusterId);
    std::vector<int> regionQuery(const std::vector<std::vector<float>>& points, int pointIdx) const;

    // KD-Tree data structures and utilities
    struct PointCloud {
        std::vector<std::vector<float>> pts;

        inline size_t kdtree_get_point_count() const { return pts.size(); }

        inline double kdtree_get_pt(const size_t idx, const size_t dim) const { return pts[idx][dim]; }

        template <class BBOX>
        bool kdtree_get_bbox(BBOX&) const { return false; }
    } cloud;

    using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<double, PointCloud>,
        PointCloud,
        -1 /* dimension = -1 means we will determine at runtime */
    >;

    KDTree* kdTree; // Use raw pointer for KDTree
};

#endif
