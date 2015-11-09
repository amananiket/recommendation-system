import pandas
import numpy
import csv
import math
from scipy.spatial.distance import cosine

NO_OF_USERS = 100

def readDataFromFile():
	ratings = pandas.read_csv('../data/ratings.dat', sep='::')
	users = pandas.read_csv('../data/users.dat', sep='::')

	return ratings, users


def calculateSimilarityMeasures(ratings, users):

	global NO_OF_USERS

	pearsonCorrMatrix = numpy.zeros((NO_OF_USERS, NO_OF_USERS))
	spearmanCorrMatrix = numpy.zeros((NO_OF_USERS, NO_OF_USERS))
	kendallCorrMatrix = numpy.zeros((NO_OF_USERS, NO_OF_USERS))
	cosineSimilarityMatrix = numpy.zeros((NO_OF_USERS, NO_OF_USERS))

	for user1ID in range(1, NO_OF_USERS):
		for user2ID in range(1, NO_OF_USERS):
			
			user1Ratings = ratings[ratings.userID == user1ID][['MovieID', 'Rating']].values
			user2Ratings = ratings[ratings.userID == user2ID][['MovieID', 'Rating']].values

			user1DataFrame = pandas.DataFrame(user1Ratings, columns= ['movie', 'rating'])
			user2DataFrame = pandas.DataFrame(user2Ratings, columns= ['movie', 'rating'])

			commonDataFrame = pandas.merge(user1DataFrame, user2DataFrame, how='inner', on=['movie'])
			commonDataFrame = commonDataFrame.drop('movie', axis = 1)

			pearsonCorr = commonDataFrame.corr(method='pearson',min_periods=1)
		   	spearmanCorr = commonDataFrame.corr(method='spearman',min_periods=1)
		   	kendallCorr = commonDataFrame.corr(method='kendall')

		   	if (math.isnan(pearsonCorr.loc['rating_x', 'rating_y'])==1):
		   		pearsonCorrMatrix[user1ID-1][user2ID-1] = 0
		   	else:
		   		pearsonCorrMatrix[user1ID-1][user2ID-1] = pearsonCorr.loc['rating_x', 'rating_y']

		   	if (math.isnan(spearmanCorr.loc['rating_x', 'rating_y'])==1):
		   		spearmanCorrMatrix[user1ID-1][user2ID-1] = 0
		   	else:
		   		spearmanCorrMatrix[user1ID-1][user2ID-1] = spearmanCorr.loc['rating_x', 'rating_y']

		   	if (math.isnan(kendallCorr.loc['rating_x', 'rating_y'])==1):
		   		kendallCorrMatrix[user1ID-1][user2ID-1] = 0
		   	else:
		   		kendallCorrMatrix[user1ID-1][user2ID-1] = kendallCorr.loc['rating_x', 'rating_y']


		   	cosineSimilarityMatrix[user1ID-1][user2ID-1] = 1 - cosine(commonDataFrame['rating_x'], commonDataFrame['rating_y'])


	return (pearsonCorrMatrix, spearmanCorrMatrix, kendallCorrMatrix, cosineSimilarityMatrix)

def main():
	ratings, users = readDataFromFile()

	PC, SC, KC, CSim = calculateSimilarityMeasures(ratings, users)

	numpy.savetxt("../data/PearsonCoefficient.csv", PC)
	numpy.savetxt("../data/SpearmanCoefficient.csv", SC)
	numpy.savetxt("../data/KendallCoefficient.csv", KC)
	numpy.savetxt("../data/CosineSimilariy.csv", CSim)


if __name__ == '__main__':
	main()