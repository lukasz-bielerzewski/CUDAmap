/**
 * Project Walkers
 * @author Dominik Belter
 */

#ifndef _CONVEX_HULL_H
#define _CONVEX_HULL_H

#include "Defs/defs.h"
#include <stack>

class ConvexHull2D{
public:
    ConvexHull2D(const std::stack<walkers::Vec3>& points){
        pointsStack = points;
    }

    ConvexHull2D(){
    }

    /// compute convex hull of a set of points
    std::vector<walkers::Vec3> compute(std::vector<walkers::Vec3>& points);

private:
    /// A utility function to find next to top in a stack
    walkers::Vec3 nextToTop(std::stack<walkers::Vec3>& stack);

    /// A utility function to swap two points
    void swap(walkers::Vec3& p1, walkers::Vec3& p2);

    /// A utility function to return square of distance between p1 and p2
    double distSq(const walkers::Vec3& p1, const walkers::Vec3& p2);

    /// To find orientation of ordered triplet (p, q, r).
    /// The function returns following values
    /// 0 --> p, q and r are collinear
    /// 1 --> Clockwise
    /// 2 --> Counterclockwise
    int orientation(const walkers::Vec3& p, const walkers::Vec3& q,
                                const walkers::Vec3& r);

    /// A function used by library function qsort() to sort an array of
    /// points with respect to the first point
    int compare(const walkers::Vec3& p1, const walkers::Vec3& p2);

    /// set of points
    std::stack<walkers::Vec3> pointsStack;
    /// first point in the set (the most on the bottom and to the right)
    walkers::Vec3 p0;
};

#endif //_CONVEX_HULL_H
