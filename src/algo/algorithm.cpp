#include "algorithm.h"
#include <algorithm>

namespace rknn_yolo {

bool Fence::contains(int x, int y) const {
    if (full_frame) return true;
    if (points.size() < 3) return false;
    bool inside = false;
    for (size_t i = 0, j = points.size() - 1; i < points.size(); j = i++) {
        const Point2i &pi = points[i];
        const Point2i &pj = points[j];
        bool intersect = ((pi.y > y) != (pj.y > y)) &&
                         (x < (pj.x - pi.x) * (y - pi.y) / (pj.y - pi.y + 1e-6f) + pi.x);
        if (intersect) inside = !inside;
    }
    return inside;
}

std::vector<cv::Point> Fence::cvPoints() const {
    std::vector<cv::Point> pts;
    pts.reserve(points.size());
    for (const auto &p : points) {
        pts.emplace_back(p.x, p.y);
    }
    return pts;
}

cv::Point Fence::center() const {
    if (points.empty()) return cv::Point(0, 0);
    long long sx = 0, sy = 0;
    for (const auto &p : points) {
        sx += p.x;
        sy += p.y;
    }
    return cv::Point(static_cast<int>(sx / points.size()), static_cast<int>(sy / points.size()));
}

} // namespace rknn_yolo
