#include <cmath> // sqrt()
#include <ctime> // time(), clock()
#include <iostream> // cout, stream
#include <fstream>
#include "common.h"

void KMeansCPU(Datapoint* data, long n, Vector2* clusters, int k){
	bool Kmeans=true;
	while(Kmeans){
		Kmeans=false;
		for(long i= 0;i<n;i++){
			//change of centroids == false
			data[i].altered=false;
			//checks distance of i point from all 3 centroids.
			for(int j =0;j<k;j++){
				if(clusters[j].distSq(data[i].p) < clusters[data[i].cluster].distSq(data[i].p)){
					//check the point is in right centroid area
					if(data[i].cluster!=j){
						data[i].cluster=j;
						//change of centroids == true
						data[i].altered=true;
					}
				}
		     }
		}
		// code to calculate new centroid
		//index for 3 clusters
		for (int j=0; j<k; j++){
			//count no. of elements under a centroid region
			long no_of_points=0;
			//index for no of points in space
			for(long i=0; i<n; i++)
			{
				if(data[i].cluster == j){
					//add x and y of each point in a particular cluster
					clusters[j].x += data[i].p.x;
					clusters[j].y += data[i].p.y;
					no_of_points++;
				}
				//if altered==true, set kmeans=true
				if(data[i].altered==true){
					Kmeans=true;
				}
			}
			//new centroid
				clusters[j].x /= no_of_points;
				clusters[j].y /= no_of_points;
		}
	}
}