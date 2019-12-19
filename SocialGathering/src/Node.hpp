#ifndef NODE_HPP
#define NODE_HPP
#include <string>

using namespace std;

class Node {
    
    public:
    string id;
    Node* prev;
    bool visited;    
    int degree;
    bool fake;
    Node(string id1, Node* prev1, bool visited1);

};
#endif