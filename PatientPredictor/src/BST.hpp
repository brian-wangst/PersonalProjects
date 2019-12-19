//
// BST.hpp
// CSE 100 Project 1
//
// Last modified by Heitor Schueroff on 01/10/2019
//

#ifndef BST_HPP
#define BST_HPP

#include <iostream>

#include "BSTIterator.hpp"
#include "BSTNode.hpp"

using namespace std;

template <typename Data>
class BST {
protected:
    // Pointer to the root of this BST, or 0 if the BST is empty.
    BSTNode<Data> *root;

    // Number of elements stored in this BST.
    unsigned int isize;

    // Height of this BST.
    unsigned int iheight;

public:
 
    // Define iterator as an aliased typename for BSTIterator<Data>.
    typedef BSTIterator<Data> iterator;

    /** 
     * Default constructor. Initializes an empty BST.
     */
    BST() : root(0), isize(0), iheight(0) {}

    /** 
     * Default destructor. Frees all memory allocated by this BST.
     */
    // TODO
    virtual ~BST() {
        deleteAll(root);
    }

    /** 
     * Inserts the given item into this BST.
     *
     * This function should use only the '<' operator when comparing
     * Data items. (do not use ==, >, <=, >=). For the reasoning
     * behind this, see the assignment writeup.
     *
     * Parameters:
     *     item Data item to insert into this BST
     *
     * Returns:
     *     true if the item was inserted as a consequence of calling
     *     this function, false otherwise (e.g. item is a duplicate).
     */
    // TODO
    // if no tree, create done
    // if less than current node, go left
    // if bigger than current node, go right 
    // base case, if at leaf node, insert 

                                            //SHOULD BE DONE!!!!1
    virtual bool insert(const Data &item) {
       if(root == 0){                       // if empty tree, create tree and initialize root.
           root = new BSTNode<Data>(item);   
           isize = isize + 1 ;
            iheight = height();
           return true;
       }
       
      BSTNode<Data>* cur = root;                              
       while(cur != 0){       // iterate down tree
          
           if(cur->data < item){
               if(cur->right == nullptr){
                   break;
               }
               cur = cur->right;                    // go right if less than data
           }
           else if (item < cur->data){
               if(cur->left == nullptr){
                   break;
               }
               cur = cur->left;                     // go left if greater than data
           }
           else{
               return false;                        // if not less than or greater than, must be equal.  Return false if dupolicate 
           }
           
       }
       
       if(cur->data < item){                        // create new node and assign to right child
           cur->right = new BSTNode<Data>(item);          // assign new node parent to cur->right
           cur->right->parent = cur;
           isize = isize + 1;
            iheight = height();
           return true;
       }
       else{
           cur -> left = new BSTNode<Data>(item);         // create new node and assign to left child
           cur->left->parent = cur;                 // assign new node parrent to cur->left
           isize = isize + 1;
            iheight = height();
           return true; 
       }

     }
     

    /**
     * Searches for the given item in this BST.
     *
     * This function should use only the '<' operator when comparing
     * Data items. (should not use ==, >, <=, >=). For the reasoning
     * behind this, see the assignment writeup.
     *
     * Parameters:
     *     item Data item to search for in this BST.
     *
     * Returns:
     *     An iterator pointing to the item if found, or pointing 
     *     past the last node in this BST if item is not found.
     */
    // TODO
    virtual iterator find(const Data &item) const {
        if(empty()){
            return 0;
        }
       BSTNode <Data> * cur = root;
       while(cur != 0){
           if(cur->data < item){
               cur = cur->right;
           }
           else if (item < cur->data){
               cur = cur->left;
           }
           else{
               return BSTIterator<Data>(cur);
           }
       }
       return BSTIterator<Data>(cur);
    }

    /** 
     * Returns the number of items currently in the BST.
     */
    // TODO                 DONE
    unsigned int size() const {
        return isize;
    }

    /** 
     * Returns the height of this BST.
     */
    // TODO
    unsigned int height() const {

       return longestPath(root);
    }
    /** 
     * Returns true if this BST is empty, false otherwise.
     */
    // TODO
    bool empty() const {
        if(root == 0){
            return true;
        }
        return false;
    }

    /** 
     * Returns an iterator pointing to the first item in the BST (not the root).
     */
    // TODO
    iterator begin() const {
        return BSTIterator<Data>(first(root));
    }

    /** 
     * Returns an iterator pointing past the last item in the BST.
     */
    iterator end() const { 
        return typename BST<Data>::iterator(0); 
    }

    /** 
     * Prints the elements in this BST in ascending order.
     */
    void inorder() const { 

        inorder(root); 
 
    }

private:
    /*
     * Find the first node in the given subtree with root curr.
     */
    static BSTNode<Data>* first(BSTNode<Data> *curr) {
        if (!curr) return 0;
        while (curr->left) {
            curr = curr->left;
        }
        return curr;
    }
    
    
     unsigned int longestPath(BSTNode<Data>* cur) const {
         if(cur ==0){
             return 0;
         }
       
      unsigned int A = longestPath(cur->left) + 1;
      unsigned int B = longestPath(cur->right) + 1;
        if(A == B){
            return A;
        }
        else if (A < B){
            return B;
        }
        return A;
    }

    /* 
     * Do an inorder traversal, printing the data in ascending order.
     *
     * You can achieve an inorder traversal recursively by following 
     * the order below.
     *
     *     recurse left - print node data - recurse right
     */
    // TODO
    // SHOULD BE DONE!!!!!!!!!
    static void inorder(BSTNode<Data> *n) {
        if(n == 0){
            return;
        }
        inorder(n->left);
        cout << n->data << endl;
        inorder(n->right);
}
        

    /* 
     * Do a postorder traversal, deleting nodes.
     *
     * You can achieve a postorder traversal recursively by following 
     * the order below.
     *
     *     recurse left - recurse right - delete node
     */
    // TODO
    static void deleteAll(BSTNode<Data> *n) {
        if(n == 0){
            return;
        }
        deleteAll(n->left);
        deleteAll(n->right);
        delete n;
        
        return;
    }
};

#endif  // BST_HPP
