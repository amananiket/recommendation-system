import pandas as pd
import numpy
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest", "AdaBoost", "Naive Bayes"]

CENTAUR = 1
EBONY = 2
TOKUGAWA = 3
ODYSSEY = 4
COSMOS = 5

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB()	]

def readDataFromFile():
	data = pd.read_csv("../data/Training.csv")

	cols_to_retain = ['Citizen ID','actual_vote','mvar_1','mvar_2','mvar_3','mvar_4','mvar_5','mvar_6','mvar_7','mvar_8','mvar_9','mvar_10','mvar_11', 'mvar_23', 'mvar_24', 'mvar_25', 'mvar_26', 'mvar_28', 'mvar_17', 'mvar_20']
	
	donationsCols = ['mvar_2','mvar_3','mvar_4','mvar_5','mvar_6']
	sharesCols = ['mvar_7','mvar_8','mvar_9','mvar_10','mvar_11']
	
	dataArray = data[cols_to_retain].values

	for i in range(0, dataArray.shape[0]):

		if (dataArray[i, 1] == 'CENTAUR'):
			dataArray[i, 1] = CENTAUR
		elif (dataArray[i, 1] == 'EBONY'):
			dataArray[i, 1] = EBONY
		elif (dataArray[i, 1] == 'TOKUGAWA'):
			dataArray[i, 1] = TOKUGAWA
		elif (dataArray[i, 1] == 'ODYSSEY'):
			dataArray[i, 1] = ODYSSEY
		elif (dataArray[i, 1] == 'COSMOS'):
			dataArray[i, 1] = COSMOS

		if (dataArray[i, 2] == 'CENTAUR'):
			dataArray[i, 2] = CENTAUR
		elif (dataArray[i, 2] == 'EBONY'):
			dataArray[i, 2] = EBONY
		elif (dataArray[i, 2] == 'TOKUGAWA'):
			dataArray[i, 2] = TOKUGAWA
		elif (dataArray[i, 2] == 'ODYSSEY'):
			dataArray[i, 2] = ODYSSEY
		elif (dataArray[i, 2] == 'COSMOS'):
			dataArray[i, 2] = COSMOS



		sumOfDonations = sum([ dataArray[i, j] for j in range(3,8)])
		#print donationsArray[i, 0], sumOfDonations

		if sumOfDonations == 0:
			for j in range(3, 8):
				dataArray[i, j] = 0
		else :

			dataArray[i,3] /= float(sumOfDonations)
			dataArray[i,4] /= sumOfDonations*1.0
			dataArray[i,5] /= sumOfDonations*1.0
			dataArray[i,6] /= sumOfDonations*1.0
			dataArray[i,7] /= sumOfDonations*1.0

		
		sumOfSocialShares = sum([ dataArray[i, j] for j in range(8, 13)])

		if sumOfSocialShares == 0:
			for j in range(8, 13):
				dataArray[i, j] = 0
		else:
			dataArray[i,8] /= sumOfSocialShares*1.0
			dataArray[i,9] /= sumOfSocialShares*1.0
			dataArray[i, 10] /= sumOfSocialShares*1.0
			dataArray[i,11] /= sumOfSocialShares*1.0
			dataArray[i,12] /= sumOfSocialShares*1.0

		sumOfRallies = 0
			
		for j in range(13, 18):
			if numpy.isnan(dataArray[i, j]):
				dataArray[i, j] = 0
			else :
				sumOfRallies += dataArray[i, j]

		if sumOfRallies != 0:

			if not numpy.isnan(dataArray[i, 13]):
				dataArray[i,13] /= sumOfRallies*1.0
			else :
				dataArray[i, 13] = 0

			if not numpy.isnan(dataArray[i, 14]):
				dataArray[i,14] /= sumOfRallies*1.0
			else :
				dataArray[i, 14] = 0

			if not numpy.isnan(dataArray[i, 15]):
				dataArray[i,15] /= sumOfRallies*1.0
			else :
				dataArray[i, 15] = 0

			if not numpy.isnan(dataArray[i, 16]):
				dataArray[i,16] /= sumOfRallies*1.0
			else :
				dataArray[i, 16] = 0
		
			if not numpy.isnan(dataArray[i, 17]):
				dataArray[i,17] /= sumOfRallies*1.0
			else :
				dataArray[i, 17] = 0

		for j in range(0, dataArray.shape[1]):
			if pd.isnull(dataArray[i, j]):
				dataArray[i, j] = 0

	df = pd.DataFrame(dataArray, columns = cols_to_retain)
	return df
	#print donationsArray


def classify(iDF):

	cols_to_retain = ['mvar_1','mvar_2','mvar_3','mvar_4','mvar_5','mvar_6','mvar_7','mvar_8','mvar_9','mvar_10','mvar_11', 'mvar_23', 'mvar_24', 'mvar_25', 'mvar_26', 'mvar_28', 'mvar_17', 'mvar_20']
	
	X = iDF[cols_to_retain]
	Y = iDF['actual_vote'].astype(int)

	Y_output = numpy.column_stack((numpy.array(Y.index.values.tolist()).flatten(), Y ))
	
	numpy.savetxt('../data/originalVotes.dat',Y_output,'%2d',delimiter=',')


	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .4)

	for name, clf in zip(names, classifiers):
		clf.fit(X_train, Y_train)
		score = clf.score(X_test, Y_test)
		Y_output = numpy.column_stack((numpy.array(Y_test.index.values.tolist()).flatten(), Y_test ))
		numpy.savetxt('../data/'+name+'-classified.dat',Y_output, '%2d',delimiter=',')
		print name, score

def main():
	inputDataFrame = readDataFromFile()

	classify(inputDataFrame)
if __name__ == '__main__':
	main()