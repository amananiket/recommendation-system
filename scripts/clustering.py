import pandas
import scipy
import numpy
import sklearn


def readFromFile():
	ratings = pandas.read_csv('../data/ratings.dat', sep='::')
	users = pandas.read_csv('../data/users.dat', sep='::')
	movies = pandas.read_csv('../data/movies.dat', sep='::')

	return ratings, users, movies

def getUserDataMatrix():

	noOfMovies = 

def main():
	ratings, users, movies = readFromFile()

if __name__ == '__main__':
	main()