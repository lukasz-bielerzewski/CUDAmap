#include "Mapping/elevationMap.h"
#include <tinyxml2.h>
#include <iostream>
#include <memory>
#include <bitset>

using namespace planner;

bool ElevationMap::load(std::string filename){
    std::ifstream file(filename);
    elevMap.clear();
    mapK2coef.clear();
    mapK3coef.clear();
    footholdCost.clear();
    cellUpdates.clear();
    colormap.clear();
    normals.clear();
    curvatures.clear();
    if (file.is_open()){ // open file
        std::string line;
        int lineNo=0; size_t rowNo=0;
        int fileColsNo=0; int fileRowsNo=0;
        while ( getline (file,line) ) { // load each line
            //std::cout << "line no " << lineNo << "\n";
            std::istringstream is(line);
            //std::cout << line << "\n";
            if (line[0]!='#'){
                if (lineNo==1){
                    is >> fileColsNo >> fileRowsNo >> rasterX >> rasterY;
                    if (fileColsNo>fileRowsNo){
                        numRows = fileColsNo; numCols = fileColsNo;
                    }
                    else{
                        numRows = fileRowsNo; numCols = fileRowsNo;
                    }
                    sizeX = rasterX*double(numCols); sizeY = rasterY*double(numRows);
                    //std::cout << "map params: " << numCols << " " << numRows << " " << sizeX << " " << sizeY << "\n";
                    elevMap = std::vector<std::vector<double>>(numRows, std::vector<double>(numCols, -0.1234));
                    mapK2coef = std::vector<std::vector<double>>(numRows, std::vector<double>(numCols, 0.0));
                    mapK3coef = std::vector<std::vector<double>>(numRows, std::vector<double>(numCols, 0.0));
                    footholdCost = std::vector<std::vector<double>>(numRows, std::vector<double>(numCols, 0.0));
                    cellUpdates = std::vector<std::vector<int>>(numRows, std::vector<int>(numCols, 0));
                    normals = std::vector<std::vector<walkers::Vec3>>(numRows, std::vector<walkers::Vec3>(numCols, walkers::Vec3(0,0,0)));
                    colormap = std::vector<std::vector<std::array<double,3>>>(numRows, std::vector<std::array<double,3>>(numCols, {{0.0,0.0,0.0}}));
                    curvatures = std::vector<std::vector<double>>(numRows-1, std::vector<double>(numCols-1, 0.0));
                    mappedCurvatures = std::vector<std::vector<double>>(numRows-1, std::vector<double>(numCols-1, 0.0));
                }
                else{
                    if (rowNo<numRows){
                        elevMap[rowNo].resize(numCols);
                        for (size_t i=0; i<(size_t)fileColsNo; i++){
                             is >> elevMap[rowNo][i];
//                             elevMap[rowNo][i]*=0.4;
//                             map[numRows-rowNo-1][i] *= 0.65;//(rowNo*0.01);
//                             if ((i<350)&&(rowNo<302)&&(rowNo>297))
//                                 elevMap[rowNo][i]=0.25;
//                             if ((i<350)&&(rowNo<402)&&(rowNo>397))
//                                 elevMap[rowNo][i]=0.25;
//                             if ((i<350)&&(rowNo<202)&&(rowNo>197))
//                                 elevMap[rowNo][i]=0.25;
                        }
                        for (size_t i=fileColsNo; i<numCols; i++){
                            elevMap[rowNo][i]=0;
                        }
                    }
                    rowNo++;
                }
            }
            lineNo++;
        }
        if (rowNo<numRows){
            for (size_t i=fileRowsNo; i<numRows; i++){
                for (size_t j=0; j<numCols; j++){
                    elevMap[i][j]=0;
                }
            }
        }
        file.close();
        setNotUpdated();
//        elevMap[0][2]=-0.2;
//        elevMap[0][1]=0.8;
        return true;
    }
    else {
        return false;
    }
}

void ElevationMap::createHeightField(simulator::RenderObjectHeightmap& heightfield){
    heightfield.x = -(sizeX/2.0);
    //heightfield.y = -((config.heightfieldSize[2] * config.heightfieldScale[2])/2.0);
    heightfield.y = -(sizeY/2.0);
    //heightfield.z = config.heightfieldSize[1];
    heightfield.z = 0;
//    heightfield.x1 = config.heightfieldScale[0];
//    heightfield.y1 = config.heightfieldScale[2];
//    heightfield.z1 = config.heightfieldScale[1];
    heightfield.scaleX = rasterX;
    heightfield.scaleY = rasterY;
    heightfield.scaleZ = 1;
    //heightfield.heightmap = config.heightmapData;
    heightfield.heightmap.resize(numRows);

    for (size_t rowNo = 0; rowNo<numRows;rowNo++){
        heightfield.heightmap[rowNo].resize(numCols);
        for (size_t colNo = 0; colNo<numCols;colNo++){
            heightfield.heightmap[rowNo][colNo]=get((int)rowNo, int(colNo));
        }
    }
    /// heightfield colors
    std::vector<std::vector<std::vector<double>>> heightmapColorData;
    std::pair<double,double> minMax;
    getMinMax(minMax);
    std::cout << "minmax " << minMax.first << ", " << minMax.second << "\n";
    for (size_t rowNo = 0; rowNo<numRows;rowNo++){
        heightmapColorData.resize(numRows);
        for (size_t colNo = 0; colNo<numCols;colNo++){
            heightmapColorData[rowNo].resize(numCols);
            std::vector<double> tempColor = {0.4, 0.4, 0.4};
            if (colorMapLoaded){
                std::array<double,3> _color = getColor((int)rowNo, int(colNo));
                tempColor.at(0) = double(_color[0])/255.0;    tempColor.at(1) = double(_color[1])/255.0;    tempColor.at(2) = double(_color[2])/255.0;
            }
            else {
                double _color[3] = {0.4, 0.4, 0.4};
                mapping::prepareHeightfieldColors(_color, colorScheme, get((int)rowNo,int(colNo)), minMax.first, minMax.second);
                tempColor.at(0) = _color[0];
                tempColor.at(1) = _color[1];
                tempColor.at(2) = _color[2];
            }
            heightmapColorData[rowNo][colNo]=tempColor;
        }
    }
    heightfield.heightmapColors = heightmapColorData;
    heightfield.type = simulator::RenderObjectType::HEIGHTFIELD;
}

///create from pcd
bool ElevationMap::createFromPCD(std::string filename){
    std::ifstream file(filename);
    elevMap.clear();
    mapK2coef.clear();
    mapK3coef.clear();
    footholdCost.clear();
    cellUpdates.clear();
    colormap.clear();
    normals.clear();
    curvatures.clear();
    mappedCurvatures.clear();
    if (file.is_open()){ // open file
        std::string line;
        int lineNo=0; size_t rowNo=0;
        numRows = 600; numCols = 600;
        rasterX=0.015; rasterY=0.015;
        sizeX = rasterX*double(numCols); sizeY = rasterY*double(numRows);
        //std::cout << "map params: " << numCols << " " << numRows << " " << sizeX << " " << sizeY << "\n";
        elevMap = std::vector<std::vector<double>>(numRows, std::vector<double>(numCols, -0.1234));
        mapK2coef = std::vector<std::vector<double>>(numRows, std::vector<double>(numCols, 0.0));
        mapK3coef = std::vector<std::vector<double>>(numRows, std::vector<double>(numCols, 0.0));
        footholdCost = std::vector<std::vector<double>>(numRows, std::vector<double>(numCols, 0.0));
        cellUpdates = std::vector<std::vector<int>>(numRows, std::vector<int>(numCols, 0));
        colormap = std::vector<std::vector<std::array<double,3>>>(numRows, std::vector<std::array<double,3>>(numCols, {{0.0,0.0,0.0}}));
        normals = std::vector<std::vector<walkers::Vec3>>(numRows, std::vector<walkers::Vec3>(numCols, walkers::Vec3(0,0,0)));
        curvatures = std::vector<std::vector<double>>(numRows-1, std::vector<double>(numCols-1, 0.0));
        mappedCurvatures = std::vector<std::vector<double>>(numRows-1, std::vector<double>(numCols-1, 0.0));
        while ( getline (file,line) ) { // load each line
            //std::cout << "line no " << lineNo << "\n";
            std::istringstream is(line);
            //std::cout << line << "\n";
            if (line[0]!='#'){
                if (lineNo>10){
                    double x,y,z,normal_x, normal_y, normal_z, curvature;
                    is >> x >> y >> z >> normal_x >> normal_y >> normal_z >> curvature;
                    walkers::Vec3 point(x,y,z);
                    update(point);
                    rowNo++;
                }
            }
            lineNo++;
        }
        file.close();
        setNotUpdated();
        return true;
    }
    else {
        return false;
    }
}

///create from pcd
bool ElevationMap::createFromCloud(const walkers::PointVec& pointCloud){
    elevMap.clear();
    mapK2coef.clear();
    mapK3coef.clear();
    footholdCost.clear();
    cellUpdates.clear();
    colormap.clear();
    normals.clear();
    curvatures.clear();
    mappedCurvatures.clear();
    numRows = 600; numCols = 600;
    rasterX=0.015; rasterY=0.015;
    sizeX = rasterX*double(numCols); sizeY = rasterY*double(numRows);
    //std::cout << "map params: " << numCols << " " << numRows << " " << sizeX << " " << sizeY << "\n";
    elevMap = std::vector<std::vector<double>>(numRows, std::vector<double>(numCols, -0.1234));
    mapK2coef = std::vector<std::vector<double>>(numRows, std::vector<double>(numCols, 0.0));
    mapK3coef = std::vector<std::vector<double>>(numRows, std::vector<double>(numCols, 0.0));
    footholdCost = std::vector<std::vector<double>>(numRows, std::vector<double>(numCols, 0.0));
    cellUpdates = std::vector<std::vector<int>>(numRows, std::vector<int>(numCols, 0));
    colormap = std::vector<std::vector<std::array<double,3>>>(numRows, std::vector<std::array<double,3>>(numCols, {{0.0,0.0,0.0}}));
    normals = std::vector<std::vector<walkers::Vec3>>(numRows, std::vector<walkers::Vec3>(numCols, walkers::Vec3(0,0,0)));
    curvatures = std::vector<std::vector<double>>(numRows-1, std::vector<double>(numCols-1, 0.0));
    mappedCurvatures = std::vector<std::vector<double>>(numRows-1, std::vector<double>(numCols-1, 0.0));
    for (const auto& point : pointCloud){
        update(point);
    }
    setNotUpdated();
    return true;
}

#ifdef BUILD_WITH_GRABBER
    ///create from pcd
    bool ElevationMap::createFromCloud(const grabber::PointCloud& cloud){
        elevMap.clear();
        mapK2coef.clear();
        mapK3coef.clear();
        footholdCost.clear();
        cellUpdates.clear();
        colormap.clear();
        normals.clear();
        numRows = 600; numCols = 600;
        rasterX=0.015; rasterY=0.015;
        sizeX = rasterX*double(numCols); sizeY = rasterY*double(numRows);
        //std::cout << "map params: " << numCols << " " << numRows << " " << sizeX << " " << sizeY << "\n";
        elevMap = std::vector<std::vector<double>>(numRows, std::vector<double>(numCols, -0.1234));
        mapK2coef = std::vector<std::vector<double>>(numRows, std::vector<double>(numCols, 0.0));
        mapK3coef = std::vector<std::vector<double>>(numRows, std::vector<double>(numCols, 0.0));
        footholdCost = std::vector<std::vector<double>>(numRows, std::vector<double>(numCols, 0.0));
        cellUpdates = std::vector<std::vector<int>>(numRows, std::vector<int>(numCols, 0));
        colormap = std::vector<std::vector<std::array<double,3>>>(numRows, std::vector<std::array<double,3>>(numCols, {{0.0,0.0,0.0}}));
        normals = std::vector<std::vector<walkers::Vec3>>(numRows, std::vector<walkers::Vec3>(numCols, walkers::Vec3(0,0,0)));
        updateFromCloud(cloud);
        return true;
    }

    ///create from pcd
    bool ElevationMap::updateFromCloud(const grabber::PointCloud& cloud){
        for (const auto& point : cloud){
            walkers::Vec3 pointXYZ(point.x, point.y, point.z);
            walkers::Vec3 color(point.r, point.g, point.b);
            update(pointXYZ, color);
        }
        return true;
    }
#endif

void ElevationMap::setNotUpdated(void){
    std::pair<double,double> minMax;
    getMinMax(minMax);
    for (auto & row: elevMap){
        for (auto & cell : row){
            if (cell==-0.1234)
                cell = minMax.first;
        }
    }
}

/// initialize mesh model from the map
void ElevationMap::getMeshModel(ObjectsMesh& objects3DS){
    ObjTypeLoader obj;
    for (size_t rowNo=0;rowNo<elevMap.size();rowNo++){
        for (size_t colNo=0;colNo<elevMap[rowNo].size();colNo++){
            Vertex vert;
            vert.vertex.x = (float)toRealX(int(colNo));
            vert.vertex.y = (float)toRealY(int(rowNo));
            vert.vertex.z = (float)get((int)rowNo, (int)colNo);
            obj.vertices.push_back(vert);
        }
    }
    for (size_t rowNo=0;rowNo<elevMap.size()-1;rowNo++){
        for (size_t colNo=0;colNo<elevMap[rowNo].size()-1;colNo++){
            PolygonType pol;
            pol.a = (rowNo*elevMap.size()+colNo); pol.b = (rowNo*elevMap.size()+colNo+1); pol.c = ((rowNo+1)*elevMap.size()+colNo); obj.polygons.push_back(pol);

//            std::cout << "vert " << pol.a << ", " << pol.b << ", " << pol.c << "\n";
//            std::cout << obj.vertices[pol.a].vertex.x << ", " << obj.vertices[pol.a].vertex.y << ", " << obj.vertices[pol.a].vertex.z << "\n";
//            std::cout << obj.vertices[pol.b].vertex.x << ", " << obj.vertices[pol.b].vertex.y << ", " << obj.vertices[pol.b].vertex.z << "\n";
//            std::cout << obj.vertices[pol.c].vertex.x << ", " << obj.vertices[pol.c].vertex.y << ", " << obj.vertices[pol.c].vertex.z << "\n";
            pol.a = (rowNo*elevMap.size()+colNo+1); pol.b = ((rowNo+1)*elevMap.size()+colNo+1); pol.c = ((rowNo+1)*elevMap.size()+colNo); obj.polygons.push_back(pol);
//            std::cout << "vert " << pol.a << ", " << pol.b << ", " << pol.c << "\n";
//            getchar();
        }
    }
    objects3DS.objects.push_back(obj);
}

/// check if the coordinates are in the range of the map
bool ElevationMap::isInRange(int row, int col) const{
    if (row<0||col<0||row>(int)elevMap.size()-1||col>(int)elevMap[row].size()-1)
        return false;
    else
        return true;
}

double ElevationMap::get(int row, int col) const {
    return ((row<int(numRows)) && (col<int(numCols)) && (row>-1) && (col>-1)) ? elevMap[row][col] : -1;
}

double ElevationMap::getFootholdCost(int row, int col) const {
    return ((row<int(numRows)) && (col<int(numCols)) && (row>-1) && (col>-1)) ? footholdCost[row][col] : -1;
}

void ElevationMap::setFootholdCost(int row, int col, double _footholdCost){
    if((row<int(numRows)) && (col<int(numCols)) && (row>-1) && (col>-1))
        footholdCost[row][col] = _footholdCost;
}

double ElevationMap::getK2Coef(int row, int col) const {
    return ((row<int(numRows)) && (col<int(numCols)) && (row>-1) && (col>-1)) ? mapK2coef[row][col] : -1;
}

double ElevationMap::getK3Coef(int row, int col) const {
    return ((row<int(numRows)) && (col<int(numCols)) && (row>-1) && (col>-1)) ? mapK3coef[row][col] : -1;
}

int ElevationMap::getClass(int row, int col) const {
    return ((row<int(mapClasses.size())) && (col<int(numCols)) && (row>-1) && (col>-1)) ? mapClasses[row][col] : -1;
}

/// get class color
std::array<double,3> ElevationMap::getClassColor(int row, int col) const {
    return ((row<int(mapClasses.size())) && (col<int(numCols)) && (row>-1) && (col>-1)) ? terrainClasses[mapClasses[row][col]].color : std::array<double,3>{{0,0,0}};
}

/// get curvature color class
std::array<double,3> ElevationMap::getCurvatureColorClass(int row, int col) const {
    std::array<double,3> color;

    /// convert mapped curvatures to 18-bit binary
    int dec = (int)mappedCurvatures[row][col];
    std::string binary = std::bitset<18>(dec).to_string();

    /// split the binary representation into 3 parts each affecting one of the RGB coefficients
    std::string red1 = binary.substr(0,3);
    std::string red2 = binary.substr(3,3);
    double alpha = (std::stoi(red1,nullptr,2))*0.1 + (std::stoi(red2,nullptr,2))*0.025;

    std::string green1 = binary.substr(6,3);
    std::string green2 = binary.substr(9,3);
    double beta = (std::stoi(green1,nullptr,2))*0.1 + (std::stoi(green2,nullptr,2))*0.025;

    std::string blue1 = binary.substr(12,3);
    std::string blue2 = binary.substr(15,3);
    double gamma = (std::stoi(blue1,nullptr,2))*0.1 + (std::stoi(blue2,nullptr,2))*0.025;

    color = {alpha,beta,gamma};
    return color;
}

/// get normals
std::vector<std::vector<walkers::Vec3>>& ElevationMap::getNormals(void){
    return normals;
}

double ElevationMap::get(double x, double y) const{
    int row, col;
    toRaster(x, y, col, row);
    return get(row, col);
}

void ElevationMap::update(const walkers::PointVec& pointCloud){
    for (const auto& point : pointCloud){
        update(point);
    }
}
///update height
void ElevationMap::update(const walkers::Vec3& point){
    int row,col;
    toRaster(point.x(), point.y(), col, row);
    if ((row>-1) && (col>-1)){
        if (row<int(elevMap.size())){
            if (col<int(elevMap.front().size())) {
                if (get(row,col)==-0.1234||get(row,col)<point.z()){
                   set(row, col, point.z());
                }
            }
        }
    }
}
///update height and color
void ElevationMap::update(const walkers::Vec3& point, const walkers::Vec3& color){
    int row,col;
    toRaster(point.x(), point.y(), col, row);
//    std::cout << point.x() << ", " <<  point.y() << ", " << point.z() << ", " <<  row << ", " <<  col << "\n";
    if ((row>-1) && (col>-1)){
        if (row<int(elevMap.size())){
            if (col<int(elevMap.front().size())) {
//                if (get(row,col)==-0.1234||get(row,col)<point.z()){
//                    set(row, col, point.z());
//                }
                set(row, col, get(row,col)+(1.0/(1.0+double(cellUpdates[row][col])))*(point.z()-get(row,col)));
                std::array<int,3> _color = {(int)color.x(), (int)color.y(), (int)color.z()};
                updateColor(row,col,_color);
            }
        }
    }
}
///update color
bool ElevationMap::updateColor(int row, int col, const std::array<int,3>& color){
    if ((row>int(numRows)) && (col>int(numCols)) && (row<0) && (col<0)) return false;
    else {
        for (size_t colorNo=0;colorNo<3;colorNo++)
            colormap[row][col][colorNo] += (1.0/(1.0+double(cellUpdates[row][col])))*(color[colorNo]-colormap[row][col][colorNo]);
        cellUpdates[row][col]++;
        return true;
    }
}

double ElevationMap::getMax(double x, double y, int area) const{
    int row, col;
    toRaster(x, y, col, row);
    double max=std::numeric_limits<double>::min();
    for (int i=-area; i<=area; i++){
        for (int j=-area; j<=area; j++) {
            if (get(row+i, col+j)>max)
                max = get(row+i, col+j);
        }
    }
    return max;
}

double ElevationMap::getMax(double x, double y, double area) const{
    int row, col;
    toRaster(x, y, col, row);
    int areaRaster = int(area / rasterX);
    double max=-10;
    for (int i=-areaRaster; i<=areaRaster; i++){
        for (int j=-areaRaster; j<=areaRaster; j++) {
            if (get(row+i, col+j)>max)
                max = get(row+i, col+j);
        }
    }
    return max;
}

bool ElevationMap::set(int row, int col, double height){
    if ((row>int(numRows)) && (col>int(numCols)) && (row<0) && (col<0)) return false;
    else { elevMap[row][col] = height; return true;}
}

bool ElevationMap::set(double x, double y, double height){
    int row, col;
    toRaster(x, y, col, row);
    if ((row<int(numRows)) && (col<int(numCols)) && (row<0) && (col<0)) return false;
    else { elevMap[row][col] = height; return true;}
}

/// compute variance
double ElevationMap::computeVariance(double x, double y, double range) const {
    double defaultHeight = get(x,y);
    double variance = 0;
    int rastersNoX = int(range/rasterX); int rastersNoY = int(range/rasterY);
    for(int i=-rastersNoX;i<=rastersNoX;i++){
        for(int j=-rastersNoY;j<=rastersNoY;j++){
            variance+= pow(get(x+i*rasterX, y+j*rasterY)-defaultHeight,2.0);
        }
    }
    return variance/((rastersNoX*2)+(rastersNoY*2)+2);
}

/// compute variance
double ElevationMap::computeVariance(walkers::Mat34 pose, double rangeX, double rangeY) const {
    double defaultHeight = get(pose(0,3),pose(1,3));
    double variance = 0;
    int rastersNoX = int(rangeX/rasterX); int rastersNoY = int(rangeY/rasterY);
    for(int i=-rastersNoX;i<=rastersNoX;i++){
        for(int j=-rastersNoY;j<=rastersNoY;j++){
            //std::cout << i << " " << j << "\n";
            walkers::Mat34 motion(walkers::Mat34::Identity());
            motion(0,3) = i*rasterX; motion(1,3) = j*rasterY;
            walkers::Mat34 currentPose = pose*motion;
            //std::cout << "get: " << currentPose(0,3) << ", " <<  currentPose(1,3) << " : " << get(currentPose(0,3), currentPose(1,3)) << std::endl;
            variance+= pow(get(currentPose(0,3), currentPose(1,3))-defaultHeight,2.0);
        }
    }
    //std::cout << "var: " << variance <<"\n";
    return variance/((rastersNoX*2)+(rastersNoY*2)+2);
}

void ElevationMap::toRaster(double x, double y, int& row, int& col) const{
    row = int(x/rasterX) + int(numCols/2)-1;
    col = -int(y/rasterY) + int(numRows/2)-1;
}

void ElevationMap::toRasterDownsampled(double x, double y, int& row, int& col, int downsampleScale) const{
    row = -int(x/rasterX) + int(numCols/2);
    col = int(y/rasterY) - int(numRows/2);
    row /= downsampleScale;
    col /= downsampleScale;
}

double ElevationMap::toRealX(int coord) const{
    return (coord-int(numCols/2)+1)*rasterX - rasterX/2.0;
}

double ElevationMap::toRealY(int coord) const{
    return -(coord-int(numRows/2)+1)*rasterY + rasterY/2.0;
}

void ElevationMap::toRealDownsampled(int coordX, int coordY, double &x, double &y, int downsampleScale) const{
    int col = coordX*downsampleScale + (downsampleScale/2);
    int row = coordY*downsampleScale + (downsampleScale/2);
    x = toRealX(col);
    y = toRealX(row);
}

std::array<double,3> ElevationMap::getColor(int row, int col) const {
    return ((row<int(numRows)) && (col<int(numCols)) && (row>-1) && (col>-1)) ? colormap[row][col] : std::array<double,3>{0.0,0.0,0.0};
}

bool ElevationMap::loadColormap(std::string filename){
    std::ifstream file(filename);
    colormap.clear();
    if (file.is_open()){ // open file
        std::string line;
        int lineNo=0; size_t rowNo=0;
        int fileColsNo=0; int fileRowsNo=0;
        while ( getline (file,line) ) { // load each line
//            std::cout << "line no " << lineNo << "\n";
            std::istringstream is(line);
//            std::cout << line << "\n";
            if (line[0]!='#'){
                if (lineNo==1){
                    is >> fileColsNo >> fileRowsNo;
                    if (fileColsNo>fileRowsNo){
                        numRows = fileColsNo; numCols = fileColsNo;
                    }
                    else{
                        numRows = fileRowsNo; numCols = fileRowsNo;
                    }
                    colormap = std::vector<std::vector<std::array<double,3>>>(numRows, std::vector<std::array<double,3>>(numCols, {{0.0,0.0,0.0}}));
                }
                else{
                    if (rowNo<numRows){
                        colormap[rowNo].resize(numCols);
                        for (size_t i=0; i<(size_t)fileColsNo; i++){
                             is >> colormap[rowNo][i][0] >> colormap[rowNo][i][1] >> colormap[rowNo][i][2];
//                             std::cout << colormap[numRows-rowNo-1][i][0] << ", " << colormap[numRows-rowNo-1][i][1] << ", " << colormap[numRows-rowNo-1][i][2] << "\n";
//                             getchar();
                        }
                        for (size_t i=fileColsNo; i<numCols; i++){
                            colormap[rowNo][i]=std::array<double,3>{{0.0,0.0,0.0}};
                        }
                    }
                    rowNo++;
                }
            }
            lineNo++;
        }
        if (rowNo<numRows){
            for (size_t i=fileRowsNo; i<numRows; i++){
                for (size_t j=0; j<numCols; j++){
                    colormap[i][j]=std::array<double,3>{{0.0,0.0,0.0}};
                }
            }
        }
        file.close();
        return true;
    }
    else {
        return false;
    }
}

bool ElevationMap::loadClasses(std::string filename){
    std::ifstream file(filename);
    mapClasses.clear();
    if (file.is_open()){ // open file
        std::string line;
        int lineNo=0; size_t rowNo=0;
        int fileColsNo=0; int fileRowsNo=0;
        while ( getline (file,line) ) { // load each line
//            std::cout << "line no " << lineNo << "\n";
            std::istringstream is(line);
//            std::cout << line << "\n";
            if (line[0]!='#'){
                if (lineNo==1){
                    is >> fileColsNo >> fileRowsNo;
                    if (fileColsNo>fileRowsNo){
                        numRows = fileColsNo; numCols = fileColsNo;
                    }
                    else{
                        numRows = fileRowsNo; numCols = fileRowsNo;
                    }
                    mapClasses = std::vector<std::vector<int>>(numRows, std::vector<int>(numCols, -1));
                }
                else{
                    if (rowNo<numRows){
                        mapClasses[rowNo].resize(numCols);
                        for (size_t i=0; i<(size_t)fileColsNo; i++){
                             is >> mapClasses[rowNo][i];
                        }
                        for (size_t i=fileColsNo; i<numCols; i++){
                            mapClasses[rowNo][i]=-1;
                        }
                    }
                    rowNo++;
                }
            }
            lineNo++;
        }
        if (rowNo<numRows){
            for (size_t i=fileRowsNo; i<numRows; i++){
                for (size_t j=0; j<numCols; j++){
                    mapClasses[i][j]=-1;
                }
            }
        }
        file.close();
        return true;
    }
    else {
        return false;
    }
}

/// Constructor
ElevationMap::ElevationMap(std::string configFilename) : name("ElevationMap"){
    setlocale(LC_NUMERIC,"C");
    tinyxml2::XMLDocument config;
    std::string filename = "../../resources/" + configFilename;
    //std::cout << "filename " << filename << "\n";
    config.LoadFile(filename.c_str());
    if (config.ErrorID())
        std::cout << "unable to load elevation map config file.\n";

    tinyxml2::XMLElement* params = config.FirstChildElement( "terrain" )->FirstChildElement( "heightfield" );
    std::string mapData(params->Attribute("filename"));
    int rotate;
    params->QueryIntAttribute("rotate",&rotate);
    double scale = 1.0;
    params->QueryDoubleAttribute("scale",&scale);

    mapData = "../../resources/" + mapData;
    std::string fileType = mapData.substr(mapData.size()-3, mapData.size()-1);
    if (fileType=="pcd"){
        createFromPCD(mapData);
    }
    else
        load(mapData);
    if (rotate!=0)
        rotateMap(elevMap, rotate);
    bool flipHorizontal(false);
    params->QueryBoolAttribute("flipHorizontal",&flipHorizontal);
    colorScheme=1;
    params->FirstChildElement("color")->QueryIntAttribute("scheme",&colorScheme);
    if (flipHorizontal)
        flipHorizontalMap(elevMap);
    if (scale!=1.0){
        scaleMap(elevMap, scale);
    }

    tinyxml2::XMLElement* colormapXML = config.FirstChildElement( "terrain" )->FirstChildElement("colormap");
    if (colormapXML != NULL) {
        colorMapLoaded = colormapXML->BoolAttribute("load");
        if (colorMapLoaded){
            std::string colormapFilename = colormapXML->Attribute("filename");
            loadColormap("../../resources/" + colormapFilename);
            std::cout << "../../resources/" + colormapFilename << "\n";
            if (rotate!=0)
                rotateMap(colormap, rotate);
            if (flipHorizontal)
                flipHorizontalMap(colormap);
        }
    }
    tinyxml2::XMLElement* mapClassesXML = config.FirstChildElement( "terrain" )->FirstChildElement("mapClasses");
    bool classesMapLoaded(false);
    if (mapClassesXML != NULL) {
        classesMapLoaded = mapClassesXML->BoolAttribute("load");
        if (classesMapLoaded){
            std::string colormapFilename = mapClassesXML->Attribute("filename");
            loadClasses("../../resources/" + colormapFilename);
            if (rotate!=0)
                rotateMap(mapClasses, rotate);
            if (flipHorizontal)
                flipHorizontalMap(mapClasses);
            terrainClasses.resize(mapClassesXML->UnsignedAttribute("classesNo"));
            for( size_t classNo = 0; classNo<mapClassesXML->UnsignedAttribute("classesNo"); classNo++){
                std::string className = "terrainClass"+std::to_string(classNo);
                tinyxml2::XMLElement* terrT = mapClassesXML->FirstChildElement(className.c_str());
                TerrainType typeT;
                typeT.id = terrT->UnsignedAttribute("id");
                typeT.name = terrT->Attribute("name");
                typeT.cost = terrT->DoubleAttribute("cost");
                typeT.color[0] = terrT->DoubleAttribute("r");
                typeT.color[1] = terrT->DoubleAttribute("g");
                typeT.color[2] = terrT->DoubleAttribute("b");
                terrainClasses[classNo] = typeT;
            }
        }
    }
    tinyxml2::XMLElement* filter = config.FirstChildElement( "terrain" )->FirstChildElement("filter");
    if (filter != NULL) {
        bool filterElevMap = filter->BoolAttribute("filterElevMap");
        bool filterClassMap = filter->BoolAttribute("filterClassMap");
        bool filterColorMap = filter->BoolAttribute("filterColorMap");
//        std::string filterType = filter->Attribute("type");
        int filterWindow = filter->IntAttribute("filterWindow");
        if (filterElevMap)
            medianFilterElev(filterWindow);
        if (filterColorMap)
            medianFilterColor(filterWindow);
        if (filterClassMap)
            medianFilterClasses(filterWindow);
    }
    //std::cout << "Map loaded\n";
}

/// rotate map
template <class T>
void ElevationMap::rotateMap(T& map, int rot){
    T mapCopy(map);
    int rowNo=0;
    for (auto & row : map){
        int colNo=0;
        for (auto & el : row){
            if (rot==90)
                mapCopy[row.size()-1-colNo][rowNo]=el;
            else if (rot==-90)
                mapCopy[colNo][row.size()-1-rowNo]=el;
            colNo++;
        }
        rowNo++;
    }
    map = mapCopy;
}

/// scale map
template <class T>
void ElevationMap::scaleMap(T& map, double scale){
    for (auto & row : map){
        for (auto & el : row){
            el *=scale;
        }
    }
}

template <class T>
void ElevationMap::flipHorizontalMap(T& map){
    T mapCopy(map);
    int rowNo=0;
    for (auto & row : map){
        int colNo=0;
        for (auto & el : row){
            mapCopy[rowNo][row.size()-1-colNo]=el;
            colNo++;
        }
        rowNo++;
    }
    map = mapCopy;
}

/// Calculate normal vector (input: vertices of the triangle, output: normal vector)
walkers::Vec3 ElevationMap::calculateNormal(const std::vector<walkers::Vec3>& vertices) const{
    walkers::Vec3 v1(vertices[0].x() - vertices[1].x(), vertices[0].y() - vertices[1].y(), vertices[0].z() - vertices[1].z());
    walkers::Vec3 v2(vertices[1].x() - vertices[2].x(), vertices[1].y() - vertices[2].y(), vertices[1].z() - vertices[2].z());

    walkers::Vec3 out(v1.y()*v2.z() - v1.z()*v2.y(), v1.z()*v2.x() - v1.x()*v2.z(), v1.x()*v2.y() - v1.y()*v2.x());
    double module = sqrt(pow(out.x(),2.0) + pow(out.y(),2.0) + pow(out.z(),2.0));
    out.x() /= module; out.y() /= module; out.z() /= module;
    return out;
}

/// export map to file
void ElevationMap::exportMap(const std::string filename) const{
    std::ofstream file(filename);
    double div=1.0;
    //file << "[X,Y]=meshgrid([" << -sizeX/2.0 << ":" << sizeX/(numCols) << ":" << sizeX/2.0 - sizeX/(numCols)<< "],[" << -sizeY/2.0 << ":" << sizeY/(numRows) << ":" << sizeY/2.0 - sizeY/(numRows) << "]);\n";
    file << "[X,Y]=meshgrid([" << -double(sizeX)/2.0 << ":" << double(sizeX)/(div*double(numCols)) << ":" << double(sizeX)/2.0 - double(sizeX)/double(div*double(numCols))<< "],[" << -double(sizeY)/2.0 << ":" << double(sizeY)/double(div*double(numRows)) << ":" << double(sizeY)/2.0 - double(sizeY)/double(div*double(numRows)) << "]);\n";
    file << "Z=[";
    /*for (auto itRow = map.begin(); itRow!=map.end(); itRow++){
        for (auto itCol = itRow->begin(); itCol!=itRow->end(); itCol++){
            file << *itCol << ", ";
        }
        file << ";\n";
    }*/
    double celldiv2x = double(sizeX)/(2*double(numCols));
    double celldiv2y = double(sizeY)/(2*double(numRows));
    double y = -sizeY/2.0+celldiv2x;
    for (size_t col=0; col<numCols; col++){
        double x = -sizeX/2.0+celldiv2y;
        for (size_t row=0; row<numRows; row++){
            file << get(x,y) << ", ";
            x += sizeX/(div*double(numCols));
        }
        file << ";\n";
        y += sizeY/(div*double(numRows));
    }

//    for (double y=-sizeY/2.0+celldiv2x; y<sizeX/2.0 - sizeX/(div*double(numCols))+celldiv2x; y=y+sizeY/(div*double(numRows))) {
//        for (double x=-double(sizeX)/2.0+celldiv2y; x<double(sizeX)/2.0 - double(sizeX)/(div*double(numCols))+celldiv2x; x=x+sizeX/(div*double(numCols))) {
//            file << get(x,y) << ", ";
//        }
//        file << ";\n";
//    }

    file << "];\n surf(X,Y,Z); colormap('autumn');\n xlabel('x'); ylabel('y'); zlabel('z');\n";
    file.close();
}

/// export map to *.dat file
void ElevationMap::exportDatMap(const std::string filename) const{
    std::ofstream file(filename);
    file << "# sizeX sizeY rasterX rasterY\n";
    file << numRows << " " << numCols << " " << rasterX << " " << rasterY << "\n";
    for (auto & row : elevMap){
        for (auto & element : row){
            file << element << " ";
        }
        file << "\n";
    }
    file.close();
}

/// export map to *.dat file
void ElevationMap::exportDatColormap(const std::string filename) const{
    std::ofstream file(filename);
    file << "# sizeX sizeY rasterX rasterY\n";
    file << numRows << " " << numCols << " " << rasterX << " " << rasterY << "\n";
    for (auto & row : colormap){
        for (auto & element : row){
            file << element[0] << " " << element[1] << " " << element[2] << " ";
        }
        file << "\n";
    }
    file.close();
}

/// compute spherical variance
void ElevationMap::normal2surface(const walkers::Mat34& pose, double rangeX, double rangeY, walkers::Vec3& normal) const{
    int numX = int((rangeX/2.0) / rasterX);
    int numY = int((rangeY/2.0) / rasterY);
    walkers::Mat34 trans; trans.setIdentity();
    std::vector< std::vector<walkers::Vec3> > pointSet;
    pointSet.resize(numX*2+1);
    for (int i=0;i<numX*2+1;i++)
        pointSet[i].resize(numY*2+1);
    int indexX=0, indexY=0;
    for (int i=-numX;i<=numX;i++){//create mesh
        indexY=0;
        for (int j=-numY;j<=numY;j++){
            trans(0,3) = rasterX*i; trans(1,3) = rasterY*j;
            walkers::Mat34 point = pose * trans;
            double height = get(point(0,3), point(1,3));
            walkers::Vec3 p3D(point(0,3), point(1,3), height);
            pointSet[indexX][indexY] = p3D;
            indexY++;
        }
        indexX++;
    }
    std::vector<walkers::Vec3> normalVectors;
    for (int i=0;i<numX*2;i++){// compute normals
        for (int j=0;j<numY*2;j++){
            walkers::Vec3 _normal;
            calculateNormal(pointSet[i][j], pointSet[i][j+1], pointSet[i+1][j+1], _normal);
            normalVectors.push_back(_normal);
            calculateNormal(pointSet[i][j], pointSet[i+1][j+1], pointSet[i+1][j], _normal);
            normalVectors.push_back(_normal);
        }
    }
    // compute average normal
    double sumX=0, sumY=0, sumZ=0;
    for (auto it=normalVectors.begin();it!=normalVectors.end();it++){
        sumX+=(*it).x(); sumY+=(*it).y(); sumZ+=(*it).z();
    }
    normal.x()=sumX/double(normalVectors.size());
    normal.y()=sumY/double(normalVectors.size());
    normal.z()=sumZ/double(normalVectors.size());
    normalizeVector(normal);
}

///calculate normal to a triangle
void ElevationMap::calculateNormal(const walkers::Vec3& p1, const walkers::Vec3& p2, const walkers::Vec3& p3, walkers::Vec3& normal) const{
    walkers::Vec3 v1(p2.x()-p1.x(), p2.y()-p1.y(), p2.z()-p1.z());
    walkers::Vec3 v2(p3.x()-p1.x(), p3.y()-p1.y(), p3.z()-p1.z());
    normal.vector() = v2.vector().cross(v1.vector());
    normalizeVector(normal);
}

/// get minMax
void ElevationMap::getMinMax(std::pair<double,double>& minMax){
    double min = std::numeric_limits<double>::max();
    double max = std::numeric_limits<double>::min();
    for (auto & row : elevMap){
        for (auto & element : row){
            if (element<min)
                min = element;
            if (element>max)
                max = element;
        }
    }
    minMax.first = min;
    minMax.second = max;
}

/// compute terrain coefs
void ElevationMap::computeK2Coef(int x, int y, int rangeX, int rangeY){
    for (int i=x-rangeX;i<=x+rangeX;i++) {
        for (int j=y-rangeY;j<=y+rangeY;j++) {
            if (i>0&&j>0&&i<(int)mapK2coef.size()&&j<(int)mapK2coef[i].size()&&mapK2coef[i][j]==0){
                double neutralHeight=get(i,j);
                for (int n=-1;n<=1;n++) {
                    for (int m=-1;m<=1;m++) {
                        mapK2coef[i][j]+=neutralHeight-get(i+n,j+m);
                    }
                }
                mapK2coef[i][j] = (mapK2coef[i][j]+0.25)/(0.5);
            }
        }
    }
}

/// compute probability of classes
bool ElevationMap::getClassProbabilities(int x, int y, int rangeX, int rangeY, int& unknown, std::vector<double>& classProb) const {
    classProb.resize(terrainClasses.size());
    std::fill(classProb.begin(), classProb.end(),0);
    size_t cellsNo=0;
    unknown=0;
    for (int rowNo=-rangeX;rowNo<rangeX;rowNo++){
        for (int colNo=-rangeY;colNo<rangeY;colNo++){
            if (getClass(x+rowNo,y+colNo)>=0){
                classProb[getClass(x+rowNo,y+colNo)]+=1.0;
                cellsNo++;
            }
            else
                unknown++;
        }
    }
    if (cellsNo==0)
        return false;
    for (size_t classNo=0; classNo<classProb.size(); classNo++) {
        classProb[classNo]=classProb[classNo]/double(cellsNo);
    }
    return true;
}

/// compute cost depenfing on the terrain type
double ElevationMap::computeTerrainCost(int x, int y, int rangeX, int rangeY){
    std::vector<double> classProb(terrainClasses.size(),0);
    int unknown=0;
    if (!getClassProbabilities(x,y, rangeX, rangeY, unknown, classProb))
        return std::numeric_limits<double>::max();
    double sum=0;
    for (size_t classNo=0; classNo<classProb.size(); classNo++) {
        sum+=classProb[classNo]*terrainClasses[classNo].cost;
    }
    if (sum==0)
        sum=10;
    return sum+unknown*100;
}

/// compute terrain coefs
void ElevationMap::computeK3Coef(int x, int y, int rangeX, int rangeY){
    for (int i=x-rangeX;i<=x+rangeX;i++) {
        for (int j=y-rangeY;j<=y+rangeY;j++) {
            if (i>0&&j>0&&i<(int)mapK3coef.size()&&j<(int)mapK3coef[i].size()&&mapK3coef[i][j]==0){
                double neutralHeight=get(i,j);
                for (int n=-1;n<=1;n++) {
                    for (int m=-1;m<=1;m++) {
                        mapK3coef[i][j]+=fabs(neutralHeight-get(i+n,j+m));
                    }
                }
                mapK3coef[i][j] = (mapK3coef[i][j])/(0.25);
            }
        }
    }
}

/// compute normal vectors for the whole map
double maxNormx = 0.0;
double minNormx = 0.0;
double maxNormy = 0.0;
double minNormy = 0.0;
double maxNormz = 0.0;
double minNormz = 0.0;
void ElevationMap::normals2surface(void){
//    double meanNormx = 0.0;
//    double meanNormy = 0.0;
//    double meanNormz = 0.0;

    for (size_t rowNo=1;rowNo<elevMap.size()-1;rowNo++){
        for (size_t colNo=1;colNo<elevMap[rowNo].size()-1;colNo++){
            double posx = toRealX((int)colNo);
            double posy = toRealY((int)rowNo);
            walkers::Vec3 normal;
            walkers::Mat34 pose(walkers::Mat34::Identity());
            pose(0,3) = posx;   pose(1,3) = posy;
            normal2surface(pose,2*rasterX+std::numeric_limits<double>::epsilon(),2*rasterY+std::numeric_limits<double>::epsilon(),normal);
            setNormal((int)rowNo, (int)colNo, normal);

            if (normal.x() < minNormx){
                minNormx = normal.x();
            }
            else if (normal.x() >= maxNormx){
                maxNormx = normal.x();
            }

            if (normal.y() < minNormy){
                minNormy = normal.y();
            }
            else if (normal.y() >= maxNormy){
                maxNormy = normal.y();
            }

            if (normal.z() < minNormz){
                minNormz = normal.z();
            }
            else if (normal.z() >= maxNormz){
                maxNormz = normal.z();
            }
            //std::cout<<"x:"<<'\t'<<normal.x()<<"y:"<<'\t'<<normal.y()<<'\t'<<"z:"<<'\t'<<normal.z()<<'\n';
//            meanNormx+=normal.x();
//            meanNormy+=normal.y();
//            meanNormz+=normal.z();
        }
    }
    /*meanNormx = meanNormx/((int(elevMap.size()))*(int(elevMap[0].size())));
    meanNormy = meanNormy/((int(elevMap.size()))*(int(elevMap[0].size())));
    meanNormz = meanNormz/((int(elevMap.size()))*(int(elevMap[0].size())));

    double varNormx = 0;
    double stdNormx = 0;

    double varNormy = 0;
    double stdNormy = 0;

    double varNormz = 0;
    double stdNormz = 0;

    for (size_t rowNo=1;rowNo<elevMap.size()-1;rowNo++){
        for (size_t colNo=1;colNo<elevMap[rowNo].size()-1;colNo++){
            walkers::Vec3 norm = getNormal((int)rowNo-1,(int)colNo-1);

            std::cout<<norm.x()<<'\t'<<norm.y()<<'\t'<<norm.z()<<'\n';

            varNormx += std::pow(norm.x()-meanNormx,2);
            varNormy += std::pow(norm.y()-meanNormy,2);
            varNormz += std::pow(norm.z()-meanNormz,2);
        }
    }

    varNormx = varNormx/((int(elevMap.size()))*(int(elevMap[0].size())));
    stdNormx = std::sqrt(varNormx);

    varNormy = varNormy/((int(elevMap.size()))*(int(elevMap[0].size())));
    stdNormy = std::sqrt(varNormy);

    varNormz = varNormz/((int(elevMap.size()))*(int(elevMap[0].size())));
    stdNormz = std::sqrt(varNormz);*/

    //std::cout<<"Standard Deviation:"<<'\t'<<stdNormx<<'\t'<<stdNormy<<'\t'<<stdNormz<<'\n';
    //std::cout<<"Minima:"<<'\t'<<minNormx<<'\t'<<minNormy<<'\t'<<minNormz<<'\n';
    //std::cout<<"Maxima:"<<'\t'<<maxNormx<<'\t'<<maxNormy<<'\t'<<maxNormz<<'\n';
}

bool ElevationMap::setNormal(int row, int col, const walkers::Vec3& normal){
    if ((row>int(numRows)) || (col>int(numCols)) || (row<0) || (col<0)) return false;
    else { normals[row][col] = normal; return true;}
}

walkers::Vec3 ElevationMap::getNormal(int row, int col) const{
    return ((row<int(numRows)) && (col<int(numCols)) && (row>-1) && (col>-1)) ? normals[row][col] : walkers::Vec3(0,0,0);
}

walkers::Vec3 ElevationMap::getNormal(double x, double y) const{
    int row, col;
    toRaster(x, y, col, row);
    return getNormal(row, col);
}

/// compute curvature for the whole map
void ElevationMap::curvatureSurface(void){
    double minCurv = curvatures[0][0];
    //int minCurvIdx[] = {0,0};
    double maxCurv = curvatures[0][0];
    //int maxCurvIdx[] = {0,0};
    //int minCurvIdx[2] = {0,0};
    //double meanCurv = 0;

    for (size_t rowNo=1;rowNo<elevMap.size()-1;rowNo++){
        for (size_t colNo=1;colNo<elevMap[rowNo].size()-1;colNo++){
            //double Z1 = get((int)rowNo-1, (int)colNo-1);
            double Z2 = get((int)rowNo-1, (int)colNo);
            //double Z3 = get((int)rowNo-1, (int)colNo+1);
            double Z4 = get((int)rowNo, (int)colNo-1);
            double Z5 = get((int)rowNo, (int)colNo);
            double Z6 = get((int)rowNo, (int)colNo+1);
            //double Z7 = get((int)rowNo+1, (int)colNo-1);
            double Z8 = get((int)rowNo+1, (int)colNo);
            //double Z9 = get((int)rowNo+1, (int)colNo+1);

            double L = rasterX;

            //double A = ((Z1 + Z3 + Z7 + Z9)/4 - (Z2 + Z4 + Z6 + Z8)/2 + Z5)/pow(L,4);
            //double B = ((Z1 + Z3 - Z7 - Z9)/4 - (Z2 - Z8)/2)/pow(L,3);
            //double C = ((-Z1 + Z3 - Z7 + Z9)/4 + (Z4 - Z6)/2)/pow(L,3);
            double D = ((Z4 + Z6)/2 - Z5)/pow(L,2);
            double E = ((Z2 + Z8)/2 - Z5)/pow(L,2);
            //double F = (-Z1 + Z3 + Z7 - Z9)/(4*pow(L,2));
            //double G = (-Z4 + Z6)/(2*L);
            //double H = (Z2 - Z8)/(2*L);
            //double I = Z5;

            double curv = -2*(D+E)*100;
            setCurvature((int)rowNo-1, (int)colNo-1, curv);

            if (curv < minCurv){
                minCurv = curv;
                //minCurvIdx[0] = (int)rowNo;
                //minCurvIdx[1] = (int)colNo;
            }
            else if (curv > maxCurv){
                maxCurv = curv;
                //maxCurvIdx[0] = (int)rowNo;
                //maxCurvIdx[1] = (int)colNo;
            }
            //meanCurv+=curv;
        }
    }
    //meanCurv = meanCurv/((int(elevMap.size()))*(int(elevMap[0].size())));

    //double varCurv = 0;
    //double stdCurv = 0;

    /*for (size_t rowNo=1;rowNo<elevMap.size()-1;rowNo++){
        for (size_t colNo=1;colNo<elevMap[rowNo].size()-1;colNo++){
            varCurv += std::pow(getCurvature((int)rowNo-1,(int)colNo-1)-meanCurv,2);
        }
    }

    varCurv = varCurv/((int(elevMap.size()))*(int(elevMap[0].size())));
    stdCurv = std::sqrt(varCurv);*/

    //std::cout<<stdCurv<<'\n';

    //std::cout<<"minCurv "<<minCurv<<" ****** "<<"maxCurv "<<maxCurv<<"\n";

    /// modify curvatures map range
    double oldMin = minCurv;
    double oldMax = maxCurv;
    double newMin = 0.0;
    double newMax = abs(maxCurv) + abs(minCurv);
    double oldRange = oldMax - oldMin;
    double newRange = newMax - newMin;

    for (size_t rowNo=1;rowNo<elevMap.size()-1;rowNo++){
        for (size_t colNo=1;colNo<elevMap[rowNo].size()-1;colNo++){
            mappedCurvatures[(int)rowNo-1][(int)colNo-1] = (((curvatures[(int)rowNo-1][(int)colNo-1] - oldMin)*newRange/oldRange) + newMin);
        }
    }
}

bool ElevationMap::setCurvature(int row, int col, const double& curv){
    if ((row>int(numRows)) || (col>int(numCols)) || (row<0) || (col<0)) return false;
    else { curvatures[row][col] = curv; return true; }
}

double ElevationMap::getCurvature(int row, int col) const{
    return ((row<int(numRows)) && (col<int(numCols)) && (row>-1) && (col>-1)) ? curvatures[row][col] : 0.0;
}

double ElevationMap::getCurvature(double x, double y) const{
    int row, col;
    toRaster(x, y, col, row);
    return getCurvature(row, col);
}

double ElevationMap::getK2(int row, int col) const{
    return ((row<int(numRows)) && (col<int(numCols)) && (row>-1) && (col>-1)) ? mapK2coef[row][col] : 0;
}

double ElevationMap::getK2(double x, double y) const{
    int row, col;
    toRaster(x, y, col, row);
    return getK2(row, col);
}

double ElevationMap::getK3(int row, int col) const{
    return ((row<int(numRows)) && (col<int(numCols)) && (row>-1) && (col>-1)) ? mapK2coef[row][col] : 0;
}

double ElevationMap::getK3(double x, double y) const{
    int row, col;
    toRaster(x, y, col, row);
    return getK2(row, col);
}

/// normalize vector
void ElevationMap::normalizeVector(walkers::Vec3& normal) const {
    double norm = normal.vector().norm();
    normal.x() /= norm;    normal.y() /= norm;    normal.z() /= norm;
}

/// median filter elevation map
void ElevationMap::medianFilterElev(int windowSize){
    std::vector<std::vector<double>> localMap = elevMap;
    for (size_t x = windowSize/2;x<elevMap.size()-windowSize/2;x++){
        for (size_t y = windowSize/2;y<elevMap[0].size()-windowSize/2;y++){
            std::vector<double> height;
            for (int wx = -windowSize/2;wx<=windowSize/2;wx++){
                for (int wy = -windowSize/2;wy<windowSize/2;wy++){
                    if (elevMap[x+wx][y+wy]!=-0.1234){
                        height.push_back(elevMap[x+wx][y+wy]);
                    }
                }
            }
            if (height.size()>0){
                std::sort(height.begin(), height.end());
                localMap[x][y]=height[height.size()/2];
            }
        }
    }
    elevMap = localMap;
}

bool comparator ( const std::pair<double,std::array<double,3>>& l, const std::pair<double,std::array<double,3>>& r)
   { return l.first < r.first; }

bool comparatorClass ( const std::pair<double,int>& l, const std::pair<double,int>& r)
   { return l.first < r.first; }

/// median filter elevation map
void ElevationMap::medianFilterColor(int windowSize){
    std::vector<std::vector<std::array<double,3>>> localMap = colormap;
    for (size_t x = windowSize/2;x<colormap.size()-windowSize/2;x++){
        for (size_t y = windowSize/2;y<colormap[0].size()-windowSize/2;y++){
            std::array<double,3> cellColor = colormap[x][y];
            if (cellColor[0]==0&&cellColor[1]==0&&cellColor[2]==0){
                std::vector<std::pair<double,std::array<double,3>>> height;
                for (int wx = -windowSize/2;wx<=windowSize/2;wx++){
                    for (int wy = -windowSize/2;wy<windowSize/2;wy++){
                        if (!(wx==0&&wy==0)){
                            std::array<double,3> cellC = colormap[x+wx][y+wy];
                            if (!(cellC[0]==0&&cellC[1]==0&&cellC[2]==0)){
                                if (elevMap[x+wx][y+wy]!=-0.1234){
                                    height.push_back(std::make_pair(elevMap[x+wx][y+wy],colormap[x+wx][y+wy]));
                                }
                            }
                        }
                    }
                }
                if (height.size()>0){
                    std::sort(height.begin(), height.end(), comparator);
                    localMap[x][y]=height[height.size()/2].second;
                }
            }
        }
    }
    colormap = localMap;
}

/// median filter elevation map
void ElevationMap::medianFilterClasses(int windowSize){
    std::vector<std::vector<int>> localMap = mapClasses;
    for (size_t x = windowSize/2;x<mapClasses.size()-windowSize/2;x++){
        for (size_t y = windowSize/2;y<mapClasses[0].size()-windowSize/2;y++){
            if (mapClasses[x][y]==-1){
                std::vector<std::pair<double,int>> classes;
                for (int wx = -windowSize/2;wx<=windowSize/2;wx++){
                    for (int wy = -windowSize/2;wy<windowSize/2;wy++){
                        if (!(wx==0&&wy==0)){
                            if (mapClasses[x+wx][y+wy]!=-1){
                                if (elevMap[x+wx][y+wy]!=-0.1234){
                                    classes.push_back(std::make_pair(elevMap[x+wx][y+wy],mapClasses[x+wx][y+wy]));
                                }
                            }
                        }
                    }
                }
                if (classes.size()>0){
                    std::sort(classes.begin(), classes.end(), comparatorClass);
                    localMap[x][y]=classes[classes.size()/2].second;
                }
            }
        }
    }
    mapClasses = localMap;
}
