import scipy.sparse as sp
import numpy as np

class Dataset(object):
    def __init__(self, path):

        self.ratingThresh = 3.99
        self.trainMatrix = self.load_rating_file_as_matrix(path + ".train.rating")
        self.testRatings = self.load_rating_file_as_list(path + ".test.rating")
        self.trainRatings = self.load_rating_file_as_list(path + ".train.rating")
        self.testNegatives = self.load_negative_file(path + ".test.negative")
        self.num_users, self.num_items = self.trainMatrix.shape

    def load_rating_file_as_list(self, filename):
        ratingList = {}
        with open(filename)as fin:
            for line in fin:
                line = line.split()
                uid = int(line[0])
                iid = int(line[1])
                r = float(line[2])
                if r > self.ratingThresh:
                    if uid in ratingList:
                        ratingList[uid].append(iid)
                    else:
                        ratingList[uid] = [iid]
        return ratingList

    def load_rating_file_as_matrix(self, filename):
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if (rating > self.ratingThresh):
                    mat[user, item] = 1.0
                line = f.readline()
        return mat

    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1: ]:
                    negatives.append(int(float(x)))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList