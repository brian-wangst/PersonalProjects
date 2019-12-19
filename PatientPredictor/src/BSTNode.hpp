//
// BSTNode.hpp
// CSE 100 Project 1
//
// Last modified by Heitor Schueroff on 01/10/2019
//

#ifndef BSTNODE_HPP
#define BSTNODE_HPP

#include <iostream>
#include <iomanip>

using namespace std;

template <typename Data>
class BSTNode {
private:

    BSTNode<Data> *sucRight(BSTNode <Data> * cur){
        if(cur->left == nullptr){     // Base case, last node on left
            return cur;
        }
      cur = sucRight(cur->left);      // go down all the way to the left
      return cur;
    }


    BSTNode<Data> *sucLeft(BSTNode <Data>*  cur){
        if(cur->right == nullptr){  // Base case, last node on the right
            return cur;
        }

        cur = sucLeft(cur->right);  // go down all the way to the right
        return cur;
    }

    BSTNode<Data> *sucAbove(BSTNode <Data> * cur){
        while(cur->parent !=0){
         if(cur->parent->left == cur){
             return cur->parent;
         }
         cur = cur->parent;
        }
        return 0;
    }


public:
    BSTNode<Data> *left;
    BSTNode<Data> *right;
    BSTNode<Data> *parent;
    Data const data;

    /**
     * Constructor that initializes a BSTNode with the given data.
     */
    BSTNode(const Data &d) : data(d) {
        left = right = parent = 0;
    }

    /**
     * Find the successor this node.
     *
     * The successor of a node is the node in the BST whose data
     * value is the next in the ascending order.
     *
     * Returns:
     *     the BSTNode that is the successor of this BSTNode,
     *     or 0 if there is none (this is the last node in the BST).
     */
    // TODO



    BSTNode<Data> *successor() {
        if(this == 0){     //if tree is empty
            return 0;
        }
       else if(this->left == 0 && this->right  == 0 && this->parent == 0){   //if only node in tree, return itself
            return this;
        }
        else if(this->right != 0){                   // look right for successor first

            return sucRight(this->right);
        }

        // else if(this->left == 0 && this->right == 0 && this->parent != 0){		//(1)this->left == 0 is not a requirement, it will cause ERROR
        else if(this->right == 0 && this->parent != 0){

            return sucAbove(this);          // if not at root note
        }

        else{                                   // look left for successor last
            return 0;
        }
    }


};


/**
 * Overload operator<< to print a BSTNode's fields to an ostream.
 */
template <typename Data>
ostream &operator<<(ostream &stm, const BSTNode<Data> &n) {
    stm << '[';
    stm << setw(10) << &n;                  // address of the BSTNode
    stm << "; p:" << setw(10) << n.parent;  // address of its parent
    stm << "; l:" << setw(10) << n.left;    // address of its left child
    stm << "; r:" << setw(10) << n.right;   // address of its right child
    stm << "; d:" << n.data;                // its data field
    stm << ']';
    return stm;
}




#endif  // BSTNODE_HPP
