#include <iostream>
#include <vector>
#include "dbscan.h"

using namespace std;

int main() {
    // Example with 512-dimensional data points
    vector<vector<double>> points = {
        vector<double>(512, 1.0),
        vector<double>(512, 1.1),
        vector<double>(512, 1.2),
        vector<double>(512, 8.0),
        vector<double>(512, 8.1),
        vector<double>(512, 8.2),
        vector<double>(512, 15.0),
        vector<double>(512, 15.1),
        vector<double>(512, 15.2)
    };

    double eps = 1.0;  // Adjust epsilon as needed
    int minPts = 5;  // Adjust minPts as needed

    DBSCAN dbscan(eps, minPts);
    dbscan.fit(points);

    const vector<int>& labels = dbscan.getLabels();

    for (size_t i = 0; i < labels.size(); ++i) {
        cout << "Point " << i << " is in cluster " << labels[i] << endl;
    }

    return 0;
}
