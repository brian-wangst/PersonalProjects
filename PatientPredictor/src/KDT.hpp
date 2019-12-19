//
// KDT.hpp
// CSE 100 Project 1
//
// Last modified by Heitor Schueroff on 01/10/2019
//

#ifndef KDT_HPP
#define KDT_HPP

#include <iostream>
#include <algorithm>
#include <math.h>
#include <limits>
#include <vector>

#include "BST.hpp"

using namespace std;

#define DELTA 0.00005

/**
 * A simple 2D Point struct with (x, y) coordinates.
 */
class Point {
public:
    double x, y;

    /** Default constructor: initializes point at (0, 0) */
    Point() : x(0.0), y(0.0) {}

    /** Constructor that initializes point to given coordinates (x, y) */
    Point(double x, double y) : x(x), y(y) {}

    /** Equals operator */
    bool operator==(const Point &other) const {
        return abs(x - other.x) < DELTA && abs(y - other.y) < DELTA;
    }

    /** Not-equals operator */
    bool operator!=(const Point &other) const {
        return abs(x - other.x) > DELTA || abs(y - other.y) > DELTA;
    }

    /** Less than operator */
    bool operator<(const Point &other) const {
        cerr << "This operator should not be called, you will need "
             << "check the x and y components each time" << endl;
        return false;
    }

    /** Returns the square of the Euclidean distance between points p1 and p2 */
    // TODO
    static double squareDistance(const Point &p1, const Point &p2) {
        return pow(p1.x - p2.x,2) + pow(p1.y - p2.y,2) * 1.0;
    }
};

/** 
 * Overload operator<< to print a Point object to an ostream. 
 */
std::ostream &operator<<(std::ostream &out, const Point &data) {
    out << "(" << data.x << ", " << data.y << ")";
    return out;
}

bool xLessThan(const Point &p1, const Point &p2) { 
    return p1.x < p2.x ; 
}

bool yLessThan(const Point &p1, const Point &p2) { 
    return p1.y < p2.y; 
}


/**
 * A 2D Tree with operations to find nearest points in 2D space.
 */
class KDT : public BST<Point> {
public:
    /** 
     * Given a vector of Point references, insert a copy into the KDT.
     *
     * Note. Because of the nature of the KDTree, a simple < operator
     *       will not be sufficient. You will need to check the x and y
     *       coordinates individually.
     * Note. Since KDT inherits from BST, you still have access to the
     *       root, isize, and iheight instance variables and should modify
     *       them accordingly.
     * Note. We have also provided a private helper method for the build
     *       function called buildSubset defined below.
     *
     * Parameters:
     *     points - a vector of points to add into a new KDT
     *
     * Returns:
     *     the number of elements in the built KDT
     */
    // TODO
    virtual unsigned int build(vector<Point> points) {
        isize = points.size();

        root = buildSubset(points,0,points.size(),0,0); // pass in end as index, what is height
        iheight = height();
        return isize;
    }

    /** 
     * Find the nearest neighbor to a given point.
     *
     * Note. THIS IS DIFFERENT THAN HOW THE PREVIOUS FIND FUNCTION WORKS
     * Note. Since KDT inherits from BST, you still have access to the
     *       root, isize, and iheight member variables.
     * Note. The same dimension ordering you used to build the tree must be used
     *       to search through it.
     * Note. We have added a private helper method for your convenience called
     *       findNNHelper defined below.
     *
     * Paremeters:
     *     p - the query point for which to find its nearest neighbor
     *
     * Returns:
     *     an iterator pointing at the nearest neighbor, or pointing at nullptr
     *     if tree is empty
     */
    // TODO
    virtual iterator findNearestNeighbor(const Point &p) const {

        double* dist = new double;
        *dist =  numeric_limits<int>::max();

        BSTNode<Point>** close  = new BSTNode<Point>*();

        if(root == 0){
            return 0;
        }

       findHelper(root, p, dist, close, 0 );
 
       BSTNode<Point> * found = &(**close);
       delete close;
       
       delete dist;
       return BSTIterator<Point>(found);
    }
    /** 
     * For the kd-tree, the find method should not be used. Use the function
     * findNearestNeighbor instead.
     */
    virtual iterator find(const Point &p) const override {
        cerr << "The KD Tree find method should not be called" << endl;
        cerr << "Use findNearestNeighbor instead" << endl;
        return 0;
    }

    /** 
     * For the kd-tree, the insert method should not be used. The tree is
     * built all at once. There is no need to add individual points.
     */
    virtual bool insert(const Point &p) override {
        cerr << "The KD Tree insert method should not be called" << endl;
        return false;
    }

private:
    /* 
     * Recursively add a subset of the array to the KD tree.
     *
     * Parameters:
     *     items - the list of points that are used to build the kd tree
     *     start - the starting index for this subsection of the tree
     *     end - the non-inclusive ending index for this subtree
     *     dimension - 0 if sorting along the x-axis, and 1 if sorting
     *                 along the y-axis
     * PostConditions: 
     *     the parent node should have its left and right children
     *     set if there are any available points and recursive
     *     calls should be made for the left and right subtree
     *     (again, if there are any remaining points).
     *
     * NOTE: 
     *     We gave you two methods: xLessThan and yLessThan. You may
     *     find them useful for the sort function from #include <algorithm>.
     */
    // TODO
    BSTNode<Point> *buildSubset(vector<Point> &points, unsigned int start, 
                                unsigned int end,
                                unsigned int dimension,
                                unsigned int height) {

                    // if(points.size() == 0){
                    //     return nullptr;         //pass in end as an index to last element
                    // }
                    if(start >= end){              //base case stop once end is > = start  
                        return 0;
                    }
                    
                    if(dimension %2 == 0){
                        sort(points.begin() + start, points.begin() + end, xLessThan); 

                    }
                    else{
                      sort(points.begin() + start, points.begin() + end, yLessThan); 

                    }
                    
                    unsigned int mid = (start + end)/2 ; //middle index or array (both even and odd)
                    
                    BSTNode<Point>* p = new BSTNode<Point>(points.at(mid)); //create node at middle of vec
                    
                    p->left = buildSubset(points, start, mid, dimension + 1, height ); //help in general stepick vs slides in implementation 
                    if(p->left != 0){
                        p->left->parent= p;
                    }
                    p->right = buildSubset(points, mid + 1, end, dimension + 1, height ); //what do i do with height
                    if(p->right !=0){
                        p->right->parent = p;
                    }

                    return p;
    
                }
                        
    /* 
     * Find the node in the subtree that is closest to the given point p
     *  
     * Parameters: 
     *     node - the root of the subtree being searched
     *     queryPoint - the point whose nearest neighbor should be found
     *     smallestSquareDistance - the smallest distance value found in
     *                              the tree so far which can be updated
     *     closestPoint - the node corresponding to the current closest
     *                    node in the tree which can be udpated
     *     dimension - flags whether the node parameter splits remaining
     *                 nodes in its subtree by the x or y dimension
     *                 (has the value of either 0 or 1)
     *
     * PostCondition:
     *     closestPoint points to the nearest neighbor
     */
    // TODO
    void findHelper(BSTNode<Point> *node, 
                      const Point &queryPoint,
                      double *smallestSquareDistance,
                      BSTNode<Point> **closestPoint,
                      unsigned int dimension) const {

            if(node == 0){            //no tree and base case
                return;
            }

 
            if(dimension %2 == 0){
                if(queryPoint.x < node->data.x){
                    //3
                    findHelper(node->left, queryPoint, smallestSquareDistance, closestPoint, dimension + 1);    //traverse to the bottom of tree
                    
                    if(*smallestSquareDistance > queryPoint.squareDistance(queryPoint, node->data)){
                     *smallestSquareDistance = queryPoint.squareDistance(queryPoint,node->data);
                        (*closestPoint) = &(*node);  
                        }
                    if(*smallestSquareDistance > queryPoint.squareDistance(Point(queryPoint.x,0), Point(node->data.x,0))){
                        findHelper(node->right, queryPoint, smallestSquareDistance, closestPoint, dimension + 1);
                    }
                }
                                                                                                                    // FIGURE OUT HOW TO ORDER THIS HSI                                                                                                           //PRETTY SURE HAVE EVERYTHIGN JUST NOT ORERED RIGHT
                else{
                    findHelper(node->right, queryPoint, smallestSquareDistance, closestPoint, dimension + 1);
                  
                    if(*smallestSquareDistance > queryPoint.squareDistance(queryPoint, node->data)){
                      *smallestSquareDistance = queryPoint.squareDistance(queryPoint, node->data);

                         (*closestPoint) = &(*node); 
                         

                    }

                    if(*smallestSquareDistance > queryPoint.squareDistance(Point(queryPoint.x,0), Point(node->data.x,0))){
                        findHelper(node->left, queryPoint, smallestSquareDistance, closestPoint, dimension + 1);
                    }
                }
            }
            
            else{
                if( queryPoint.y < node->data.y){
                    findHelper(node->left, queryPoint, smallestSquareDistance, closestPoint, dimension + 1);
                    if(*smallestSquareDistance > queryPoint.squareDistance(queryPoint, node->data)){
                      *smallestSquareDistance = queryPoint.squareDistance(queryPoint, node->data);
                         (*closestPoint) = &(*node);   
                        }
                     if(*smallestSquareDistance > queryPoint.squareDistance(Point(0,queryPoint.y), Point(0,node->data.y))){
                        findHelper(node->right, queryPoint, smallestSquareDistance, closestPoint, dimension + 1);
                     }
                }
                else{
                    findHelper(node->right, queryPoint, smallestSquareDistance, closestPoint, dimension + 1);
                    
                    if(*smallestSquareDistance > queryPoint.squareDistance(queryPoint, node->data)){
                      *smallestSquareDistance = queryPoint.squareDistance(queryPoint, node->data);
                         (*closestPoint) = &(*node);
                    }
                     if(*smallestSquareDistance > queryPoint.squareDistance(Point(0,queryPoint.y), Point(0,node->data.y))){
                        findHelper(node->left, queryPoint, smallestSquareDistance, closestPoint, dimension + 1);
                     }
                }
            }
                      }

        // while(node->left !=0 && node->right != 0){
        //     if(dimension %2 == 0){            //compare x ONLY
        //          if(queryPoint.x < node->data.x){
        //              node = node->left;
        //             }
        //          else{
        //              node = node->right;
        //             }
        //         }
                
        //     else{
        //         if(queryPoint.y < node->data.y){    //compare y ONLY
        //             node = node->left;
        //             }
        //          else{
        //             node = node->left;
        //             }
        //         }
        //         dimension = dimension + 1;  //updated dimension everytime I go down the tree
        //     
    
    
    
    

                      
                     
                     
   

};

#endif  // KDT_HPP
