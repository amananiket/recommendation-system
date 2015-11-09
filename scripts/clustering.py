import pandas
import scipy
import numpy
from sklearn.cluster import KMeans, AgglomerativeClustering
NO_OF_USERS = 100
NO_OF_CLUSTERS = 10

def readFromFile():
	ratings = pandas.read_csv('../data/ratings.dat', sep='::')
	users = pandas.read_csv('../data/users.dat', sep='::')
	movies = pandas.read_csv('../data/movies.dat', sep='::')

	return ratings, users, movies

def getUserDataMatrix(ratings, users, movies):

	global NO_OF_USERS

	moviesSize = movies.shape[0]
	moviesIndices = movies['MovieID']
	noOfMovies = moviesIndices[moviesSize-1]
	print noOfMovies
	noOfColumns = noOfMovies + 3

	userDataMatrix = numpy.zeros((NO_OF_USERS, noOfColumns))

	for user in range(1, NO_OF_USERS):

		userGenders = users['Gender']
		userAges = users['Age']
		userProfessions = users['Profession']

		if (userGenders[user-1] == "M"):			
			userDataMatrix[user-1, 0] = 1
		else:
			userDataMatrix[user-1, 0] = 0


		userDataMatrix[user-1, 1] = userAges[user-1]
		userDataMatrix[user-1, 2] = userProfessions[user-1]

	ratedUsers = ratings['userID']
	ratedMovies = ratings['MovieID']
	ratingPoints = ratings['Rating']

	for rating in range(0, ratings.shape[0]):

		if (ratedUsers[rating] in range(1, NO_OF_USERS)):
			userDataMatrix[ratedUsers[rating]-1, 2 + ratedMovies[rating]] = ratingPoints[rating]


	return userDataMatrix

def main():
	ratings, users, movies = readFromFile()
	global NO_OF_CLUSTERS, NO_OF_USERS

	userDataMatrix =  getUserDataMatrix(ratings, users, movies)

	kmeansModel = KMeans(n_clusters = NO_OF_CLUSTERS, init='k-means++')
	agglomerativeClusteringModel = AgglomerativeClustering(n_clusters = NO_OF_CLUSTERS, affinity='euclidean')

	kmeansPredictedClusters = kmeansModel.fit_predict(userDataMatrix)

	aggClusteringPredictedClusters = agglomerativeClusteringModel.fit_predict(userDataMatrix)
	
	with open("../data/UserClustersKMeans.txt", "w") as KMeansOutFile:
		for i in range(1, NO_OF_USERS):
			KMeansOutFile.write("\t".join([str(i), str(kmeansPredictedClusters[i-1])]) + "\n")


	with open("../data/UserClustersAgglomerativeClustering.txt", "w") as aggClusOutFile:
		for i in range(1, NO_OF_USERS):
			aggClusOutFile.write("\t".join([str(i), str(aggClusteringPredictedClusters[i-1])]) + "\n")



if __name__ == '__main__':
	main()

