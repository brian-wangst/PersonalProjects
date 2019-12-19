#include "Node.hpp"
#include <string>

using namespace std;


   Node::Node(string id, Node* prev, bool visited) : id(id), prev(prev), visited(visited) {}

   
   