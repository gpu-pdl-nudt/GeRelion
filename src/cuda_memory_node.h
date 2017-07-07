/***************************************************************************
 *
 * Author: "Sjors H.W. Scheres"
 * MRC Laboratory of Molecular Biology
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/
/***************************************************************************
 *
 * Authors:     Carlos Oscar S. Sorzano (coss@cnb.csic.es)
 *
 * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
 * 02111-1307  USA
 *
 *  All comments concerning this program package may be sent to the
 *  e-mail address 'xmipp@cnb.csic.es'
 ***************************************************************************/

#ifndef CUDA_MEMORY_NODE_H_
#define CUDA_MEMORY_NODE_H_
#include <cuda_runtime.h>
#include <signal.h>

#include <string.h>
#include <iomanip>
#include "src/math_function.h"
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

static void HandleError( cudaError_t err, const char *file, int line )
{

    if (err != cudaSuccess)
    {
    	fprintf(stderr, "ERROR: %s in %s at line %d (error-code %d)\n", cudaGetErrorString( err ), file, line, err );
		fflush(stdout);
		raise(SIGSEGV);
    }
}
//templat<typename T>
class cuda_memory_node
{
public:
    // The array itself
    cuda_memory_node* pPre;
    cuda_memory_node* pNext;
    size_t size;
   unsigned char  *p_dev;
    bool  is_busy;

    cuda_memory_node():
		pPre(NULL), pNext(NULL), size(0),
		p_dev(NULL), is_busy(false)
			{} ;
   cuda_memory_node(size_t totalsize)
   {
	  pPre = NULL;
	  pNext =NULL;
	  size = totalsize;
	  HANDLE_ERROR(cudaMalloc( (void**) &p_dev, totalsize));
	  is_busy = false;
   }
		
    cuda_memory_node(size_t totalsize, void *pPtr):
		pPre(NULL), pNext(NULL), size(totalsize),
		p_dev((unsigned char*) pPtr), is_busy(false)
			{} ;
   ~cuda_memory_node()
   	{
		pPre = NULL;
		pNext = NULL;
		p_dev = NULL;
		size = 0;
		is_busy = false;
   };

   public:
   	unsigned char * getPoint (){return p_dev;}
    size_t getNodeSize() { return size;}
	bool isBusy() {return is_busy;}
	size_t setNodeSize( size_t allocatesize) { size = allocatesize;}
	unsigned char *getFisrtsuitableNode();
	void release_node()
	{
		is_busy = false;
	}
	
	inline
	cuda_memory_node *_getFirstSuitedNode(cuda_memory_node *first, size_t size)
	{
		cuda_memory_node *a = first;
		//If not the last and too small or not free go to next allocation region
		while (a != NULL && ( a->size <= size || ! a->is_busy ) )
			a = a->pNext;

		return a;
	}

	void freeOneNode( cuda_memory_node *a, cuda_memory_node*first)
	{
		a->is_busy = false;

			//Previous neighbor is free, concatenate
			if ( a->pPre != NULL && !a->pPre->is_busy)
			{
				//Resize and set pointer
				a->size += a->pPre->size;
				a->p_dev = a->pPre->p_dev;

				//Fetch secondary neighbor
				cuda_memory_node *ppL = a->pPre->pPre;

				//Remove primary neighbor
				if (ppL == NULL) //If the previous is first in chain
					first = a;
				else
					ppL->pNext = a;

				delete a-> pPre;

				//Attach secondary neighbor
				a->pPre = ppL;
			}

			//Next neighbor is free, concatenate
			if ( a->pNext != NULL && !a->pNext->is_busy)
			{
				//Resize and set pointer
				a->size += a->pNext->size;

				//Fetch secondary neighbor
				cuda_memory_node*nnL = a->pNext->pNext;

				//Remove primary neighbor
				if (nnL != NULL)
					nnL->pPre = a;
				delete a->pNext;

				//Attach secondary neighbor
				a->pNext = nnL;
			}
	}
		
	
    void freeIdealNodes(cuda_memory_node *first)
    {
		cuda_memory_node *next = first;
		cuda_memory_node *curr;

		while (next != NULL)
		{
			curr = next;
			next = curr->pNext;

			if (! curr->is_busy )
			{
					freeOneNode(curr, first);
					next = first; //List modified, restart
			}
		
		}
	}
	cuda_memory_node * create_node( cuda_memory_node *first, size_t size)
	{
		freeIdealNodes(first);
		cuda_memory_node  *a  = _getFirstSuitedNode(first,  size);
		
		cuda_memory_node *newNode(NULL);
		if (a == NULL)
		{
			std::cout<<"Out of memory" << " in File "<<__FILE__ << " at line " <<__LINE__ << std::endl;
			raise(SIGSEGV);
		}
		if(a->size == size)
		{
			a->is_busy= true;
			newNode = a;
		}
		else 
		{
			//Setup new pointer
			newNode = new cuda_memory_node();
			newNode->pNext = a;
			newNode->p_dev= a->p_dev;
			newNode->size = size;
			newNode->is_busy = true;

			//Modify old pointer
			a->p_dev = &(a->p_dev[size]);
			a->size -= size;

				//Insert new allocation region into chain
				if(a->pPre == NULL) //If the first allocation region
					first = newNode;
				else
					a->pPre->pNext = newNode;
				newNode->pPre = a->pPre;
				newNode->pNext = a;
				a->pPre = newNode;
		}
		cudaMemset( newNode->p_dev, 0, size * sizeof(unsigned char));

		return newNode;
	}
};


#endif
   
