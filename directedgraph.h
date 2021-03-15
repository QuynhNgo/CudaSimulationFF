/* File: directedgraph.h
 * Author: Quynh Ngo
 * Created at 7:50 pm on October 13th, 2015
 */

#ifndef DIRECTEDGRAPH_H
#define DIRECTEDGRAPH_H

#include <iostream>
#include <fstream>
#include <vector>

#include "utils.h"


//  Lowercase: name for class
//  Uppercase: name for global variables
// _Uppercase: name for local variables, which includes input variables

/* @des an edge of a directed network
 * @member BeginNode is the name of the node from which the edge goes out
 * @member EndNode is the name of the node from which the edge goes to
 * @memberFunc operator== is to check if two edges are the same
 */
struct edge
{
	int BeginNode;
	int EndNode;
	bool operator==(edge _MyEdge)
	{
		if( (BeginNode == _MyEdge.BeginNode) && (EndNode == _MyEdge.EndNode) )
			return true;
		else return false;
	}
};


/*
 * @des a directed graph that does basic IO, File IO, 
 *      and basic computation on a directed graph
 */
class directedgraph
{

public:

	// basic constructors
	directedgraph();
	directedgraph(const directedgraph &_TmpCopyGraph);
	// read directly from file
	directedgraph(std::string _Filename ); 
	
	// destructor with no heap memory so far to delete
	~directedgraph();


	// overloading operator=
	void operator =(const directedgraph &_TmpGraph);

	/* Procedures, functions of graph; */
	
	// to  get the In_Degree_List;
	std::vector<std::vector<int>> getInList() const;	
	// get the number of node;
	int		getNumberNode() const; 
	void	setNumberNode(int tmpVar);


	/* Parser */

	/* 
	 * @des from link list, the output is gonna be the outindegree
	 * @param[in] _FileName is the name of the source file
	 */
	void loadGraphFromFile(std::string _Filename ); 
	
	
	/*
	 * @des export undirected graph in form of link-list
	 * @param[in] _FileName is the name of the file, to which the graph is exported to
	 */
	void printGraph2File(std::string _FileName) const; 
	
private:

	// the coresponding list to node i will be including 
	// all node go to i;
	std::vector<std::vector<int>> InList; 
	// Number of nodes of the graph
	int NumberNode;

};

#endif