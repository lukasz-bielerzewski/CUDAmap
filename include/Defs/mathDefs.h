/** @file mathDefs.h
* @author: Dominik Belter
* Walkers Math definitions
*
*/

#ifndef MATHDEFS_H_INCLUDED
#define MATHDEFS_H_INCLUDED

#include "eigen3.h"
#include "Defs/defs.h"

/// walkers name space
namespace walkers {

    class Plane3D{
        public:
        Plane3D() : A(0.0), B(0.0), C(0.0), D(0.0){
        }

        /// print plane model
        void print(void);

        /// compute angle between points
        double angleBetweenPlanes(const Plane3D& plane);

        /// distance between plane and point
        double distance(const walkers::Vec3& point) const{
            return fabs(A*point.x()+B*point.y()+C*point.z()+D)/sqrt(pow(A,2.0)+pow(B,2.0)+pow(C,2.0));
        }

        /// extract plane from three random points
        void fromPoints(const std::vector<walkers::Vec3>& points);

        /// coefficients
        double A,B,C,D;

        /// point on the plane
        walkers::Vec3 x0;
    };

    class Triangle3D{
        public:
        Triangle3D(){
        }
        /// print
        void print(void);

        /// vertices
        std::array<walkers::Vec3,3> vertices;

        /// normal vector
        walkers::Vec3 normal;
    };

    class Line3D{
        public:
        Line3D() : A(0.0), B(0.0), C(0.0), x0(0,0,0){
        }
        /// print
        void print(void);

        /// line from Mat34 (rotation mat and point)
        void fromPose(const walkers::Mat34& pose);

        /// direction
        double A,B,C;
        /// point on the line
        walkers::Vec3 x0;
    };

    class Arrow3D{
        public:
        Arrow3D() : x0(0,0,0), x1(1.0,0,0){
        }
        Arrow3D(const walkers::Vec3& _x0, const walkers::Vec3& _x1) : x0(_x0), x1(_x1){
        }
        /// print
        void print(void);

        /// start
        walkers::Vec3 x0;
        /// end
        walkers::Vec3 x1;
    };

    template <class T> class BSpline{
    public:
        BSpline(){}
        BSpline(size_t _order) : order(_order){
            initialize();
        }

        /// initialize
        void initialize(void){
            M = Eigen::MatrixXd(order,order);
            for (size_t s=0;s<order-1;s++){
                for (size_t n=0;n<order-1;n++){
                    M(s,n)=blendingElement(s,n);
                }
            }
            createKnotVector(int(order), int(controlPoints.size()));
    //        std::cout << "M\n" << M << "\n";
        }

        ///returns the knot value at the passed in index
        double knot(int knotIdx){
            return double(knotVector.at(knotIdx));
        }

        ///calculate the whole knot vector
        void createKnotVector(int curveOrderK, int numControlPoints){
            knotVector.clear();
//            knotVector = std::vector<int>({0, 0, 1, 2, 3, 4, 5, 6, 6});
            for(int count = 0; count < curveOrderK + numControlPoints; count++){
                int id = count;
                if (count>=numControlPoints)
                    id = numControlPoints - 1;
                knotVector.push_back(id);
            }
        }

        /// update control points
        void setControlPoints(const std::vector<T>& _controlPoints){
            controlPoints = _controlPoints;
            createKnotVector(int(order), int(controlPoints.size()));
        }
        /// get control points
        std::vector<T> getControlPoints(void){
            return controlPoints;
        }

        /// calculate weight for ith control point
        double calculateWeightForPoint(int i, int k, int controlPointNo, double t){
            //bottom of the recursive call
            if(k==1){
                if(t >= knot(i) && t < knot(i+1))
                    return 1;
                else
                    return 0;
            }

            double subweightA = 0;
            double subweightB = 0;

            if((knot(i+k-1)-knot(i))!=0)
                subweightA = (t-knot(i))/(knot(i+k-1)-knot(i)) * calculateWeightForPoint(i, k-1, controlPointNo, t);
            if((knot(i+k)-knot(i+1))!=0)
                subweightB = (knot(i+k) - t) / (knot(i+k)-knot(i+1)) * calculateWeightForPoint(i+1, k-1, controlPointNo, t);

            return subweightA + subweightB;
        }

        /// t should be in range <0,1>
        virtual T compValue(double t) = 0;
        /// t should be in range 1,...,pointsNo
        virtual T compValueRecursive(double t) = 0;

        /// order of the B-spline
        size_t order;
        /// knot vector
        std::vector<int> knotVector;
        /// blending matrix
        Eigen::MatrixXd M;

    private:
        /// compute binomial coefficient
        int binomialCoeff(int s, int k) {
            return factorial(k)/(factorial(s)*factorial(k-s));
        }

        /// compute element of the blending matrix
        double blendingElement(size_t s, size_t n){
            double sum=0;
            for (size_t l=s;l<order;l++){
                sum+=pow(-1.0,double(l-s))*
                        double(binomialCoeff(int(l-s),int(order)))*
                        pow(double(order-1-l),double(order-1-n));
            }
            return (double(binomialCoeff(int(n),int(order-1)))/double(walkers::factorial(int(order-1))))*sum;
        }

    protected:
        /// control points
        std::vector<T> controlPoints;
    };

    class BSpline3D : public BSpline<walkers::Vec3>{
        public:
        BSpline3D(){}
        BSpline3D(size_t _order) : BSpline(_order){
            initialize();
        }

        /// t should be in range <0,1>
        walkers::Vec3 compValue(double t);
        /// t should be in range 1,...,pointsNo
        walkers::Vec3 compValueRecursive(double t);

    private:
    };

    class BSplineSE3 : public BSpline<walkers::Mat34>{
        public:
        BSplineSE3(){}
        BSplineSE3(size_t _order) : BSpline(_order){
            initialize();
        }

        /// t should be in range <0,1>
        walkers::Mat34 compValue(double t);
        /// t should be in range 1,...,pointsNo
        walkers::Mat34 compValueRecursive(double t);
        /// get trajectory
        std::vector<walkers::Mat34> getTrajectory(size_t pointsNo);

//        double interValueAbs(Eigen::Vector4d values, double u);
//        double interValueCum(Eigen::Vector4d values, double u);
//        Eigen::Vector3d interSO3Cum(Eigen::Matrix<double, 3, 4> & values, double u);
//        Eigen::Matrix4d interTSO3Cum(Eigen::Matrix4d & Ta, Eigen::Matrix4d & Tb, Eigen::Matrix4d & Tc, Eigen::Matrix4d & Td, double u);

    private:
    };

    /// balance matrix
    void balance_matrix(const walkers::Mat33& A, walkers::Mat33& Aprime, walkers::Mat33& D);

}

#endif // MATHDEFS_H_INCLUDED
