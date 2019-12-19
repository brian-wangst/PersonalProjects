#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <unordered_map>
#include <queue>
#include <limits>
#include <cfloat>
#include <stack>
#include <vector>
#include "Node.hpp"
#include "Graph.hpp"


using namespace std;

Graph::Graph() {}

Graph::~Graph() {
  destructor();
}


 
/* Read in relationships from an inputfile to create a graph */

  bool Graph::loadFromFile(const char* in_filename){
  ifstream infile(in_filename);
  
  while (infile) {
    string s;
    if (!getline(infile, s)) break;

    istringstream ss(s);
    vector<string> record;
  
    while (ss) {
      string s;
      if (!getline(ss, s, ' ')) break;
      record.push_back(s);    
      built.insert(s);    //use unordered set to store Vertex's built without duplicates
    }

    if (record.size() != 2) {
      continue;
    }
    
    //4 cases
    //1). Both exist, create new nodes for at key and pushi into vector.  Than create connection between both
    //2). Both don't exist, create connection between both
    //3). First exists, second doesnt.  Create second and create connection
    //4). Second exists, first doesnt.  Create first and create connection
    
      unsigned int record0 = map[record.at(0)].size();    //Let's me know if ID was created
      unsigned int record1 = map[record.at(1)].size();
    
      if ((record0 > 0) && (record1== 0)){   //first exists but second doesntt
        Node* vertex = new Node(record.at(1),0,false);
        vertex->fake = false;
        map[record.at(1)].push_back(vertex);
        addEdge(record.at(0), record.at(1));
      }
    
    
      else if ((record0 > 0) && (record1 > 0)){  // both nodes exist
        addEdge( record.at(0), record.at(1));
      }
    
      else if((record0 == 0) && (record1 == 0)) {    //both don't exist
        Node* vertex = new Node(record.at(0),0,false);      
        vertex->fake = false;
        Node* vertex2 = new Node(record.at(1),0,false);
        vertex2->fake = false;
        map[record.at(0)].push_back(vertex);
        map[record.at(1)].push_back(vertex2);
        addEdge( record.at(0), record.at(1));
      }
  

      else {
        Node* vertex = new Node(record.at(0), 0, false);  //first doens't exist but second does
        vertex->fake = false;
        map[record.at(0)].push_back(vertex);
        addEdge( record.at(0), record.at(1));
      }
    
    }
    
        if (!infile.eof()) {
      cerr << "Failed to read " << in_filename << "!\n";
      return false;
    }

    infile.close();

    return true;
  }






//return the path from from -> to
  string Graph::pathfinder(Node* from, Node* to) {
   
    if(from->id == to->id){                      //if passed in the same id, checks if there is connection too itself
      return from->id;
    }
    
    queue <Node*> q;
    
    vector <Node*> seen;                //vector that holds strings that have been visited: used to reset the graph after each search
    
    Node* cur = map[from->id].at(0);        
    cur->visited = true;                  //initialize first node to visited 
    seen.push_back(cur);
    q.push(cur);                          //begin bfs by adding in start node to queue
  
    while(!q.empty()){
      
      Node* cur = q.front();
      q.pop();
    
      if(cur->id == to->id){                //condition checks if the node was found
      string output = printPath(cur);
      resetGraph(seen); 
      return output;
      }
  
      for(unsigned int i = 1; i < map[cur->id].size(); ++i){     //adding any unsearched nodes into the queue
        if(map[cur->id].at(i)->visited == false){
          Node* neighbor = map[cur->id].at(i);
          neighbor->visited = true;
          neighbor->prev = cur;
          q.push(neighbor);
          string location = neighbor->id;
          seen.push_back(map[location].at(0));       //any time a node visited, need to push into our seen vector
        }
      } 
      
    }
  
  resetGraph(seen);  
  return "\n";
 }


  //1). Pop if fake node    
  //2). pop if done node
  //3). push if replacement node
  
  string Graph::socialgathering(vector<pair< string,int>> &core, const int& k) {
    vector<Node*> heap;
    unordered_set<string>::iterator it;
    
    for(it = built.begin(); it != built.end(); ++it){   
    map[*it].at(0)->degree = map[*it].size() - 1;
    heap.push_back(map[*it].at(0));
    }
    
    int size = heap.size();
    for(int i = 0; i < size; ++i){
      vector<Node*>::iterator min = min_element(heap.begin(), heap.end(), NodePtrComp());
      
      pair <string, int> coreVal;
      coreVal.first = (*min)->id;
      coreVal.second = (*min)->degree;
      core.push_back(coreVal);
      string ID = (*min)->id;
      
      
      for(int j = 1; j < map[ID].size(); ++j){
        if(map[ID].at(j)->degree > (*min)->degree){
          map[ID].at(j)->degree = map[ID].at(j)->degree - 1;

        }
      }
      heap.erase(min);
    }
    return printK(core, k);
  }
  
  
  string Graph::printK(vector <pair<string, int>> & core, const int & k){

    vector <int> invite;
    
    string output;
    for(int i = 0; i < core.size(); ++i){
      if(core.at(i).second >= k){

      invite.push_back(stoi(core.at(i).first));
       }
    }
    
    sort(invite.begin(), invite.end());
    
    for(int i = 0; i < invite.size(); ++i){

      output = output + to_string(invite.at(i)) + "\n";
    }
    
    return output;
  }
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

  //Function: Creates connection in unordered map to point toward allocated memry at index 0.  Each connection represents a neighbor
  void Graph::addEdge(const string&  node1, const string&  node2){

      Node* connection = map[node1].at(0);  //creates connection pointing to node1

      map[node2].push_back(connection);      //puhses node into unordered map at key node2
      
      Node* connection2 = map[node2].at(0);   //creates connection pointing to node 2

      map[node1].push_back(connection2);    //pushes node unoredered map at key node1
  
  }

  //Function: Reset member variables prev and visited to 0 and false for additional pathfinder searches on nodes traversed
  void Graph::resetGraph(vector <Node*>& seen){ 
    for(int i = 0; i< seen.size(); ++i){
      seen.at(i)->prev = 0;
      seen.at(i)->visited = false;
    }
  }

  //Funciton: Delete all alllcated memory in the grpah
  void Graph::destructor(){
      unordered_set<string>::iterator it;
      for(it = built.begin(); it != built.end(); ++it){   
        delete map[*it].at(0);                // all allocated memory at index 0
      }
  }
  
  //Parameters: Node that points to the final ID to be found
  //Returns: The path from starting ID to the final ID
  string Graph::printPath(Node* cur){
    stack <Node*> path;
    string output = "";
    while(cur != 0){
      path.push(cur);
      cur = cur->prev;
    }
    while(!path.empty()){
      output = output + path.top()->id;
      path.pop();
      if(!path.empty()){
        output = output + " ";
      }
    }
    return output;
  }