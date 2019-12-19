#ifndef GRAPH_HPP
#define GRAPH_HPP
#include <string>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "Node.hpp"


#include <iostream>

using namespace std;

class Graph {
 protected:

 

 public:
  Graph();


  ~Graph(void);
  
   unordered_map <string, vector <Node*>> map;
   unordered_set <string> built;
   
   class NodePtrComp {
    public:
    bool operator()(Node* lhs, Node* rhs) const{
        return lhs->degree < rhs->degree;
    }
};
 
 
  //MAYBE ADD SOME MORE METHODS HERE SO AS TO ANSWER QUESTIONS IN YOUR PA
	
  /* YOU CAN MODIFY THIS IF YOU LIKE , in_filename : THE INPUT FILENAME */

  bool loadFromFile(const char* in_filename);

 string pathfinder(Node* from, Node* to);
 
 void resetGraph(vector <Node*>& seen);
    
  string socialgathering( vector<pair<string, int>> & core, const int& k);
  
  string printK(vector<pair<string, int>> & core, const int & k);
  
  void addEdge( const string&, const string&);
  
  void destructor();
  
  string printPath(Node* cur);

};

#endif  // GRAPH_HPP
