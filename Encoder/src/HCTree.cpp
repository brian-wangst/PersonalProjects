#include <stack>
#include <queue>

#include "HCTree.hpp"

/**
 * Destructor for HCTree
 */
HCTree::~HCTree() {
    deleteAll(root);
}

/** Use the Huffman algorithm to build a Huffman coding tree.
 *  PRECONDITION: freqs is a vector of ints, such that freqs[i] is
 *  the frequency of occurrence of byte i in the message.
 *  POSTCONDITION:  root points to the root of the tree,
 *  and leaves[i] points to the leaf node containing byte i.
 */
void HCTree::build(const vector<int>& freqs) {
    if(root != 0){
        deleteAll(root);
    }
    
    std::priority_queue<HCNode*, std::vector<HCNode*>, HCNodePtrComp> pq;
    for(int i = 0; i < freqs.size(); ++i){
        if(freqs.at(i) != 0){           //if count is 0, its garbage, don't keep 
        HCNode * p = new HCNode(freqs.at(i),i,0,0,0);
        pq.push(p);
        leaves.push_back(p);
        }
    }
    
    if(pq.size() == 0){
        return;
    }

    
    while(pq.size() != 1){
        
        HCNode* small1 = pq.top();      //take two smallest counts of priority queue 
        pq.pop();
        HCNode* small2 = pq.top();
        pq.pop(); 
        
        HCNode* newPar = new HCNode(small1->count + small2->count,0,small1,small2,0);
        
        if(small1->symbol < small2->symbol){
            newPar->symbol = small2->symbol;
        }
        else{
            newPar->symbol = small1->symbol;
        }
        
        small1->p = newPar;
        small2->p= newPar;
        
        
        pq.push(newPar);

    }
    root = pq.top();
}

/** Write to the given ostream
 *  the sequence of bits (as ASCII) coding the given symbol.
 *  PRECONDITION: build() has been called, to create the coding
 *  tree, and initialize root pointer and leaves vector.
 */
void HCTree::encode(byte symbol, ostream& out) const {
    stack <char> stack;                                //getting to leaf node
    HCNode* cur;
    
    for(int i = 0; i < leaves.size(); ++i){
        if(leaves.at(i) != nullptr && leaves.at(i)->symbol == symbol){
            cur = leaves.at(i);
            break;
        }
    }
    
    while(cur!= root){             //storing left and right into stack
        if(cur->p->c0 == cur){
            stack.push('0');
        }
        else{
            stack.push('1');
        }
        cur = cur->p;
    }
    
    while(!stack.empty()){              //outputting to ostream 
        out << stack.top();
        stack.pop();
    }
    
}

/** Return the symbol coded in the next sequence of bits (represented as 
 *  ASCII text) from the istream.
 *  PRECONDITION: build() has been called, to create the coding
 *  tree, and initialize root pointer and leaves vector.
 */
byte HCTree::decode(istream& in) const {
    if(root == 0){
        return 0;
    }
    if(root->c0 == 0 && root->c1 == 0){
        return root->symbol;
    }
    HCNode* cur = root;
    int count = 0;
    char c;
    byte value;
    while(in >> c){
        if(c == '0'){
            cur = cur->c0;
            if(cur->c0 == 0 && cur->c1 == 0){
                break;
            }
        }
        else{
            cur=cur->c1;
            if(cur->c1 == 0 && cur->c1 == 0){
                break;
            }
        }
    }
     
     value = cur->symbol;
     return value;

    // return 0;  // TODO (checkpoint)
}

/** Write to the given BitOutputStream
 *  the sequence of bits coding the given symbol.
 *  PRECONDITION: build() has been called, to create the coding
 *  tree, and initialize root pointer and leaves vector.
 */
void HCTree::encode(byte symbol, BitOutputStream& out) const {
    stack <bool> stack;                                //getting to leaf node
    HCNode* cur;
    
    for(int i = 0; i < leaves.size(); ++i){
        if(leaves.at(i) != nullptr && leaves.at(i)->symbol == symbol){
            cur = leaves.at(i);
            break;
        }
    }
    
    while(cur!= root){             //storing left and right into stack
        if(cur->p->c0 == cur){
            stack.push(false);
        }
        else{
            stack.push(true);
        }
        cur = cur->p;
    }
    
    
    while(!stack.empty()){              //outputting to ostream 
        out.writeBit(stack.top());
        stack.pop();
    }
}

/** Return symbol coded in the next sequence of bits from the stream.
 *  PRECONDITION: build() has been called, to create the coding
 *  tree, and initialize root pointer and leaves vector.
 */
byte HCTree::decode(BitInputStream& in) const {
   if(root == 0){
        return 0;
    }
    if(root->c0 == 0 && root->c1 == 0){
        return root->symbol;
    }
    HCNode* cur = root;
    int count = 0;
    char c;
    byte value;
    while(cur != 0){
        int bit = in.readBit();  //read the byte 
        if(bit == 0){
            cur = cur->c0;
            if(cur->c0 == 0 && cur->c1 == 0){
                break;
            }
        }
        else{
            cur=cur->c1;
            if(cur->c1 == 0 && cur->c1 == 0){
                break;
            }
        }
    }
     
     value = cur->symbol;
     return value;
}

/**
 * Print the contents of a tree
 */
void HCTree::printTree() const {
    cout << "=== PRINT TREE BEGIN ===" << endl;
    printTreeHelper(root);
    cout << "=== PRINT TREE END =====" << endl;
}

/**
 * Recursive helper function for printTree
 */
void HCTree::printTreeHelper(HCNode * node, string indent) const {
    if (node == nullptr) {
        cout << indent << "nullptr" << endl;
        return;
    }

    cout << indent << *node << endl;
    if (node->c0 != nullptr || node->c1 != nullptr) {
        printTreeHelper(node->c0, indent + "  ");
        printTreeHelper(node->c1, indent + "  ");
    }
}

void HCTree::deleteAll(HCNode* cur){
    if(cur == 0){
        return;
    }
    deleteAll(cur->c0);
    deleteAll(cur->c1);
    delete cur;
    return;
}