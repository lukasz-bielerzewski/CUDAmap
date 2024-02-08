/** @file mapping.h
 *
 * Mapping class
 *
 */

#ifndef _MAPPING_H_
#define _MAPPING_H_

#include "Defs/planner_defs.h"
#include "Defs/simulator_defs.h"
#include "Defs/mapping_defs.h"
#ifdef BUILD_WITH_GRABBER
#include "Defs/grabber_defs.h"
#endif
#include "Utilities/objectsMesh.h"

/// 2.5D elevation map
class ElevationMap {
    public:
        /// set of maps
        typedef std::vector<ElevationMap> Seq;

        /// Terrain type
        class TerrainType {
            public:
                TerrainType(){
                }
                /// class id
                size_t id;
                /// class name
                std::string name;
                /// transition cost
                double cost;
                /// visualization color
                std::array<double,3> color;
        };

        /// map size
        double sizeX, sizeY;

        /// map size
        size_t numRows, numCols;

        /// raster size
        double rasterX, rasterY;

        ///  constructor
        ElevationMap(void) : sizeX(1.0), sizeY(1.0), numRows(1), numCols(1),
            rasterX(double(sizeX)/double(numRows)),
            rasterY(double(sizeY)/double(numCols)),
            elevMap(numRows, std::vector<double>(numCols, 0.0)),
            mapK2coef(numRows, std::vector<double>(numCols, 0.0)),
            mapK3coef(numRows, std::vector<double>(numCols, 0.0)),
            footholdCost(numRows, std::vector<double>(numCols, 0.0)),
            name("ElevationMap"),
            colorScheme(1),
            colorMapLoaded(false) {
        }

        ///  constructor
        ElevationMap(size_t _numRows, size_t _numCols, double _sizeX, double _sizeY) :
            sizeX(_sizeX),
            sizeY(_sizeY),
            numRows(_numRows),
            numCols(_numCols),
            rasterX(double(_sizeX)/double(_numRows)),
            rasterY(double(_sizeY)/double(_numCols)),
            elevMap(numRows, std::vector<double>(numCols, 0.0)),
            mapK2coef(numRows, std::vector<double>(numCols, 0.0)),
            mapK3coef(numRows, std::vector<double>(numCols, 0.0)),
            footholdCost(numRows, std::vector<double>(numCols, 0.0)),
            name("ElevationMap"),
            colorScheme(1),
            colorMapLoaded(false){
        }

        /// Constructor
        ElevationMap(std::string configFilename);

        const std::string& getName() const { return name; }

        /// check if the coordinates are in the range of the map
        bool isInRange(int row, int col) const;

        double get(int row, int col) const;

        double getFootholdCost(int row, int col) const;

        void setFootholdCost(int row, int col, double _footholdCost);

        double getK2Coef(int row, int col) const;

        double getK3Coef(int row, int col) const;

        std::array<double,3> getColor(int row, int col) const;

        /// get class color
        std::array<double,3> getClassColor(int row, int col) const;

        /// get curvature color
        std::array<double,3> getCurvatureColorClass(int row, int col) const;

        int getClass(int row, int col) const;

        /// compute probability of classes
        bool getClassProbabilities(int x, int y, int rangeX, int rangeY, int& unknown, std::vector<double>& classProb) const;

        void createHeightField(simulator::RenderObjectHeightmap& heightfield);

        double get(double x, double y) const;

        /// get number of rows
        size_t getNumRows(void) const {
            return numRows;
        }

        /// get number of cols
        size_t getNumCols(void) const {
            return numCols;
        }

        void update(const walkers::PointVec& pointCloud);
        ///create from pcd
        bool createFromCloud(const walkers::PointVec& pointCloud);
#ifdef BUILD_WITH_GRABBER
        ///create from pcd
        bool createFromCloud(const grabber::PointCloud& cloud);
        ///update from pcd
        bool updateFromCloud(const grabber::PointCloud& cloud);
#endif

        void toRaster(double x, double y, int& row, int& col) const;

        void toRasterDownsampled(double x, double y, int& row, int& col, int downsampleScale) const;

        double toRealX(int coord) const;

        double toRealY(int coord) const;

        void toRealDownsampled(int coordX, int coordY, double &x, double &y, int downsampleScale) const;

        double getMax(double x, double y, int area) const;

        double getMax(double x, double y, double area) const;

        bool set(int row, int col, double height);

        bool setNormal(int row, int col, const walkers::Vec3& normal);

        walkers::Vec3 getNormal(int row, int col) const;

        walkers::Vec3 getNormal(double x, double y) const;

        bool setCurvature(int row, int col, const double& curv);

        double getCurvature(int row, int col) const;

        double getCurvature(double x, double y) const;

        double getK2(int row, int col) const;

        double getK2(double x, double y) const;

        double getK3(int row, int col) const;

        double getK3(double x, double y) const;

        bool set(double x, double y, double height);

        bool updateColor(int row, int col, const std::array<int,3>& color);

        /// compute variance
        double computeVariance(double x, double y, double range) const;

        /// compute variance
        double computeVariance(walkers::Mat34 pose, double rangeX, double rangeY) const;

        /// Load elevation map from file
        bool load(std::string filename);

        /// Load colormap from file
        bool loadColormap(std::string filename);

        /// Load classes from file
        bool loadClasses(std::string filename);

        /// Calculate normal vector (input: vertices of the triangle, output: normal vector)
        walkers::Vec3 calculateNormal(const std::vector<walkers::Vec3>& vertices) const;

        /// export map to file
        void exportMap(const std::string filename) const;

        /// export map to *.dat file
        void exportDatColormap(const std::string filename) const;

        /// compute spherical variance
        void normal2surface(const walkers::Mat34& pose, double rangeX, double rangeY, walkers::Vec3& normal) const;

        /// compute normal vectors for the whole map
        void normals2surface(void);

        /// compute curvatures for the whole map
        void curvatureSurface(void);

        /// get data
        inline std::vector<std::vector<double>>& getMap() {return elevMap;}

        /// get minMax
        void getMinMax(std::pair<double,double>& minMax);

        /// compute terrain coefs
        void computeK2Coef(int x, int y, int rangeX, int rangeY);

        /// compute terrain coefs
        void computeK3Coef(int x, int y, int rangeX, int rangeY);

        /// compute cost depenfing on the terrain type
        double computeTerrainCost(int x, int y, int rangeX, int rangeY);

        /// median filter elevation map
        void medianFilterElev(int windowSize);

        /// median filter color
        void medianFilterColor(int windowSize);

        /// median filter color
        void medianFilterClasses(int windowSize);

        /// export map to *.dat file
        void exportDatMap(const std::string filename) const;

        /// set cells which haven't been updated
        void setNotUpdated(void);

        /// initialize mesh model from the map
        void getMeshModel(ObjectsMesh& objects3DS);

        /// get normals
        std::vector<std::vector<walkers::Vec3>>& getNormals(void);

    private:
        /// 2.5D map
        std::vector<std::vector<double>> elevMap;
        /// K2 coefs map
        std::vector<std::vector<double>> mapK2coef;
        /// K3 coefs map
        std::vector<std::vector<double>> mapK3coef;
        /// K3 coefs map
        std::vector<std::vector<double>> footholdCost;
        /// normal vectors map
        std::vector<std::vector<walkers::Vec3>> normals;
        /// colormap
        std::vector<std::vector<std::array<double,3>>> colormap;
        /// cell updates
        std::vector<std::vector<int>> cellUpdates;
        /// 2.5D map
        std::vector<std::vector<int>> mapClasses;
        /// name of the map
        std::string name;
        /// classes
        std::vector<TerrainType> terrainClasses;
        /// color scheme
        int colorScheme;
        /// load colormap
        bool colorMapLoaded;
        /// 2.5D curvatures map
        std::vector<std::vector<double>> curvatures;
        /// curvatures map with modified range (0 to max value)
        std::vector<std::vector<double>> mappedCurvatures;

        ///calculate normal to a triangle
        void calculateNormal(const walkers::Vec3& p1, const walkers::Vec3& p2, const walkers::Vec3& p3, walkers::Vec3& normal) const;
        /// normalize vector
        void normalizeVector(walkers::Vec3& normal) const;
        /// rotate map
        template <class T>
        void rotateMap(T& map, int rot);
        /// rotate map
        template <class T>
        void scaleMap(T& map, double scale);
        /// flip horizontal
        template <class T>
        void flipHorizontalMap(T& map);
        ///update height
        void update(const walkers::Vec3& point);
        ///update height and color
        void update(const walkers::Vec3& point, const walkers::Vec3& color);
        ///create from pcd
        bool createFromPCD(std::string filename);
};


#endif // _MAPPING_H_
