/** @file weightedGraph.h
 *
 * Weighted Graph
 * \author Dominik Belter
 */

#ifndef _WEIGHTEDGRAPH_H_
#define _WEIGHTEDGRAPH_H_

#include "../Defs/defs.h"
#include <set>
#include <iostream>
#include <fstream>

namespace walkers {

    template <class T>
    class WeightedVertex;
    template <class T>
    using VertexSet = std::list<std::shared_ptr<WeightedVertex<T>>>;
    template <class T>
    using VertexIDMap = std::unordered_map<int, std::shared_ptr<WeightedVertex<T>>>;

    template <class T>
    class WeightedEdge;
    template <class T>
    using EdgeSet = std::list<std::shared_ptr<WeightedEdge<T>>>;

    /// WeightedGraph class
    template <class T>
    class WeightedGraph {
    public:
//        typedef std::list<std::shared_ptr<WeightedEdge<T>>> EdgeSet;
//        template <class N>
//        using EdgeSet = std::list<std::shared_ptr<WeightedEdge<N>>>;
//        typedef std::list<std::shared_ptr<WeightedVertex<T>>> VertexSet;
//        template <class N>
//        using VertexSet = std::list<std::shared_ptr<WeightedVertex<N>>>;
//        typedef std::unordered_map<int, std::shared_ptr<WeightedVertex<T>>> VertexIDMap;
//        template <class N>
//        using VertexIDMap = std::unordered_map<int, std::shared_ptr<WeightedVertex<N>>>;
        //typedef std::pair<std::shared_ptr<WeightedVertex>,std::shared_ptr<WeightedVertex>> VerticesPair;
        /// add vertex
        bool addVertex(int vertexId){
            auto got = vertices.find(vertexId);
            if ( got != vertices.end() ){
                return false;
            }
            else {
                vertices.insert(std::make_pair(vertexId,std::shared_ptr<WeightedVertex<T>>(new WeightedVertex<T>(vertexId))));
                return true;
            }
        }

        /// add edge
        bool addEdge(const WeightedEdge<T>& e){
            edges.insert(edges.end(),std::shared_ptr<WeightedEdge<T>>(new WeightedEdge(e)));
            std::pair<int,int> verts = e.getVertices();

            //std::cout << "vertices " << verts.first << ", " << verts.second << "\n";
            if (!addEdgeToVertex(verts.first,edges.back())) return false;
            if (!addEdgeToVertex(verts.second,edges.back())) return false;
            //std::cout << "covisibility graph size: " << vertices.size() << "\n";
            //std::cout << "edges: " << edges.size() << " \n";
            return true; // DB: Check that!
        }

        /// find neighbouring vertices
        void findNeighbouringNodes(int id, double covisibilityThr, std::set<int>& verticesIds){
            WeightedVertex<T>* vert = vertices.at(id).get();
        //    WeightedGraph::EdgeSet edgesNeigh;
        //    vert->getEdges(edgesNeigh);
            verticesIds.clear();
            std::vector<std::pair<int,double>> neighbours;
            vert->getNeighbours(neighbours);
            for (auto neighbour : neighbours){
                if (neighbour.second>covisibilityThr)
                    verticesIds.insert(neighbour.first);
            }
        }

        /// add edge to vertex
        bool addEdgeToVertex(int vertexId, const std::shared_ptr<WeightedEdge<T>>& e){
            auto got = vertices.find(vertexId);
            if ( got == vertices.end() ){
                return false;
            }
            else {
                got->second->addEdge(e);
                return true;
            }
        }

        /// save to file
        void save2file(std::string filename){
            std::cout << "Saving covisibility graph\n";
            std::ofstream graphfile(filename);
            graphfile << vertices.size() << " ";
            for (const auto& vert : vertices){
                graphfile << vert.first << " ";
                graphfile << vert.second->getId() << " ";
                EdgeSet<T> weightedEdges;
                vert.second->getEdges(weightedEdges);
                graphfile << weightedEdges.size() << " ";
                for (const auto& e : weightedEdges){
                    graphfile << e->getWeight() << " " << e->getVertices().first << " " << e->getVertices().second << " ";
        //            graphfile << e->getId() << " " << e->getWeight() << " " << e->getVertices().first << " " << e->getVertices().second << " ";
                }
            }
        //    graphfile << edges.size() << " ";
        //    for (const auto& e : edges)
        //        graphfile << e->getId() << " " << e->getWeight() << " " << e->getVertices().first << " " << e->getVertices().second << " ";
        //    graphfile.close();
            graphfile << "Done\n";
        }

        /// load
        void load(std::string filename){
            std::cout << "Loading covisibility graph\n";
            std::ifstream graphfile(filename);
            size_t vertSize;
            graphfile >> vertSize;
            for (size_t vertNo=0; vertNo<vertSize; vertNo++){
                int vertId;
                graphfile >> vertId;
                this->addVertex(vertId);
                size_t edgesNo;
                graphfile >> edgesNo;
                for (size_t edgeNo=0; edgeNo<edgesNo; edgeNo++){
                    double weight; std::pair<int,int> vertIds;
                    graphfile >> weight >> vertIds.first >> vertIds.second;
                    walkers::WeightedEdge<T> edge(weight, std::make_pair(vertIds.first, vertIds.second));
                    addEdge(edge);
                }
            }
            std::cout << "Done\n";
        }

    protected:
        /// Vertices
        VertexIDMap<T> vertices;

        /// Edges
        EdgeSet<T> edges;
    };

    template <class T>
    class WeightedEdge {
    public:
        inline WeightedEdge() : id (idCounter++){}
        inline WeightedEdge(double _weight, std::pair<int,int> _vertIds) :
            id(idCounter++), weight(_weight), vertIds(_vertIds){}
        virtual ~WeightedEdge(){}
        /// returns the id
        int getId() const {return id;}
        /// returns vertices
        inline std::pair<int,int> getVertices(void) const {return vertIds;}
        /// returns weight
        inline double getWeight(void) const {return weight;}
    protected :
        /// id
        int id;
        /// cost
        double weight;
        /// vertices
        std::pair<int,int> vertIds;
    private:
        static inline int idCounter = 0;
    };

    template <class T>
    class WeightedVertex {
    public:
        inline WeightedVertex() {}
        inline WeightedVertex(int vertexId) : id(vertexId){}
        virtual ~WeightedVertex(){}

        /// returns the id
        int getId() const {return id;}

        /// add edge
        void addEdge(const std::shared_ptr<WeightedEdge<T>>& e) {
            edges.insert(edges.end(),e);
        }

        /// get edges
        void getEdges(EdgeSet<T>& weightEdges) {
            weightEdges=edges;
        }

        /// get neighbours
        void getNeighbours(std::vector<std::pair<int,double>>& neighbours){
            for (auto edge : edges){
                int vertId;
                std::pair<int,int> vertices = edge->getVertices();
                if (vertices.first!=id)
                    vertId = vertices.first;
                else if (vertices.second!=id)
                    vertId = vertices.second;
                else
                    throw std::runtime_error(std::string("Something is wrong with edges in weighted graph\n"));
                neighbours.push_back(std::make_pair(vertId,edge->getWeight()));
            }
        }

    protected :
        /// id
        int id;
        /// value
        T value;
        /// edges
        EdgeSet<T> edges;
    };
}

#endif // _WEIGHTEDGRAPH_H_
