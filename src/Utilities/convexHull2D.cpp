#include "Utilities/convexHull2D.h"
#include <stack>
#include <algorithm>
#include <functional>
#include <iostream>
// inspired by https://www.geeksforgeeks.org/convex-hull-set-2-graham-scan/

/// A utility function to find next to top in a stack
walkers::Vec3 ConvexHull2D::nextToTop(std::stack<walkers::Vec3>& stack){
    walkers::Vec3 p = stack.top();
    stack.pop();
    walkers::Vec3 res = stack.top();
    stack.push(p);
    return res;
}

/// A utility function to swap two points
void ConvexHull2D::swap(walkers::Vec3& p1, walkers::Vec3& p2) {
    walkers::Vec3 temp = p1;
    p1 = p2;
    p2 = temp;
}

/// A utility function to return square of distance between p1 and p2
double ConvexHull2D::distSq(const walkers::Vec3& p1, const walkers::Vec3& p2) {
    return pow(p1.x() - p2.x(),2.0) +
            pow(p1.y() - p2.y(),2.0);
}

/// To find orientation of ordered triplet (p, q, r).
// The function returns following values
// 0 --> p, q and r are collinear
// 1 --> Clockwise
// 2 --> Counterclockwise
int ConvexHull2D::orientation(const walkers::Vec3& p, const walkers::Vec3& q,
                            const walkers::Vec3& r) {
    double val = (q.y() - p.y()) * (r.x() - q.x()) -
            (q.x() - p.x()) * (r.y() - q.y());
    if (val == 0.0) return 0;  // collinear
    return (val > 0.0)? 1: 2; // clock or counterclock wise
}

/// A function used by library function qsort() to sort an array of
/// points with respect to the first point
int ConvexHull2D::compare(const walkers::Vec3& p1, const walkers::Vec3& p2) {
    // Find orientation
    double o = orientation(p0, p1, p2);
    if (o == 0.0)
        return (distSq(p0, p2) >= distSq(p0, p1))? -1 : 1;

    return (o == 2)? -1: 1;
}

/// compute convex hull of a set of points
std::vector<walkers::Vec3> ConvexHull2D::compute(std::vector<walkers::Vec3>& points) {
    // Find the bottommost point
    double ymin = points[0].y();
    size_t min = 0;
    size_t pointNo = 0;
    for (const auto& point : points){
        double y = point.y();

        // Pick the bottom-most or choose the left
        // most point in case of tie
        if ((y < ymin) || (ymin == y && point.x() < points[min].x())){
            ymin = point.y();
            min = pointNo;
        }
        pointNo++;
    }

    // Place the bottom-most point at first position
    swap(points[0], points[min]);

    // Sort n-1 points with respect to the first point.
    // A point p1 comes before p2 in sorted output if p2
    // has larger polar angle (in counterclockwise
    // direction) than p1
    p0 = points[0];

    std::sort(points.begin()+1, points.end(), [this](const walkers::Vec3& a, const walkers::Vec3& b){
        if (this->compare(a, b)<0)
            return true;
        else
            return false;
    });

    // If two or more points make same angle with p0,
    // Remove all but the one that is farthest from p0
    // Remember that, in above sorting, our criteria was
    // to keep the farthest point at the end when more than
    // one points have same angle.
    size_t m = 1; // Initialize size of modified array
    for (pointNo=1; pointNo<points.size(); pointNo++){
        // Keep removing i while angle of i and i+1 is same
        // with respect to p0
        while (pointNo < points.size()-1 && orientation(p0, points[pointNo], points[pointNo+1]) == 0)
            pointNo++;

        points[m] = points[pointNo];
        m++;  // Update size of modified array
    }

    // If modified array of points has less than 3 points,
    // convex hull is not possible
    if (m < 3) return std::vector<walkers::Vec3>();

    // Create an empty stack and push first three points
    // to it.
    std::stack<walkers::Vec3> stack;
    stack.push(points[0]);
    stack.push(points[1]);
    stack.push(points[2]);

    // Process remaining n-3 points
    for (size_t i = 3; i < m; i++){
        // Keep removing top while the angle formed by
        // points next-to-top, top, and points[i] makes
        // a non-left turn
        while (stack.size()>1 && orientation(nextToTop(stack), stack.top(), points[i]) != 2)
            stack.pop();
        stack.push(points[i]);
    }

    pointsStack = stack;
    std::vector<walkers::Vec3> v;
    while (!stack.empty()) {
        v.push_back(stack.top());
        stack.pop();
    }

    return v;
}
