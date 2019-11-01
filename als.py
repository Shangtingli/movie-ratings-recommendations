import sys,os
from pyspark import SparkContext
from numpy.random import rand
import numpy as np
import time
#########HyperParameters#########
LAMBDA = 0.01
ITERATIONS = 10
F =5
partitions = 2
########Util Functions##########

##Functions to calculate the matrix with the least squared error 
##Reference: http://alexhwilliams.info/itsneuronalblog/2018/02/26/censored-lstsq/
def censored_lstsq(A, B, M):
    """Solves least squares problem subject to missing data.

    Note: uses a broadcasted solve for speed.

    Args
    ----
    A (ndarray) : m x r matrix
    B (ndarray) : m x n matrix
    M (ndarray) : m x n binary matrix (zeros indicate missing values)

    Returns
    -------
    X (ndarray) : r x n matrix that minimizes norm(M*(AX - B))
    """

    # Note: we should check A is full rank but we won't bother...

    # if B is a vector, simply drop out corresponding rows in A
    if B.ndim == 1 or B.shape[1] == 1:
        return np.linalg.lstsq(A[np.ravel(M)], B[np.ravel(M)])[0]

    # else solve via tensor representation
    rhs = np.dot(A.T, M * B).T[:,:,None] # n x r x 1 tensor
    T = np.matmul(A.T[None,:,:], M.T[:,:,None] * A[None,:,:]) # n x r x r tensor
    return np.squeeze(np.linalg.solve(T, rhs)).T # transpose to get r x n



#Construct Baskets RDD from Text File
def getBasketsRdd():
    file = sc.textFile(INPUT, minPartitions=partitions)
    ratings = file.map(splitRow).filter(lambda x: x[0] != 'userId' and x[1] != 'movieId').map(lambda x: (int(x[0]),(int(x[1]),float(x[2]))))
    return ratings

#Split a row in text file, util function
#userId,movieId
def splitRow(row):
    words = row.split(",")
    return (words[0],words[1],words[2])


def constructMatrix(userToRatings_rdd,movie_translation_b):
    temp = userToRatings_rdd.map(lambda x:changeMatrixRow(x,movie_translation_b))\
        .sortBy(lambda x: x[0])
    matrix = temp.map(lambda x: x[1]).reduce(lambda x,y: merge(x,y))
    return matrix

def merge(row1,row2):
    return np.concatenate((row1,row2),axis = 0)

##########ALS Functions##########
def rmse(R, ms, us):
    temp =  ms * us.T
    count = 0
    summ = 0.0
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if np.isnan(R[i,j]):
                continue
            summ += np.power(R[i,j] - temp[i,j],2)
            count+=1
    
    return np.sqrt(summ/float(count))

def update(i, mat, ratings):
    # print(mat.shape,ratings.shape)
    uu = mat.shape[0]
    ff = mat.shape[1]
    # XtX = mat.T * mat
    temp = ratings[i, :].reshape(1,len(ratings[i,:]))
    # Xty = mat.T * temp.T
    mask = generateMask(temp)
    # res = np.linalg.solve(XtX, Xty)
    res = censored_lstsq(mat, temp.T, mask.T)
    return res

def generateMask(y):
    # print("Xty Shape:",Xty.shape)
    mask = np.zeros((y.shape),dtype=bool)
    # print("Mask Shape:" ,mask.shape)
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if np.isnan(y[i,j]):
                mask[i,j] = 0
            else:
                mask[i,j] = 1
    return mask
########################################
def changeMatrixRow(user_ratings,movie_translation_b):
    userId = user_ratings[0]
    ratings = user_ratings[1]
    return_row = np.empty((1,NUM_MOVIES),dtype = float)*(np.nan)
    for (movieId,rating) in ratings:
        index = movie_translation_b.value[movieId]
        return_row[0][index] = rating
    return (userId,return_row)
    
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('You also need to pass in the input and output file')
        print('spark-submit als.py ratings.csv als/predictions.csv')
        sys.exit(1)
    if 'als' not in os.listdir('.'):
        os.mkdir('als')

    global start_time
    start_time = time.time()
    global INPUT
    INPUT = sys.argv[1]
    global OUTPUT
    OUTPUT = sys.argv[2]
    sc = SparkContext(appName = "Homework 3")
    ratings_rdd = getBasketsRdd()
    # print(ratings_rdd.take(4))
    userToRatings_rdd = ratings_rdd.groupByKey()
    # print(userToRatings_rdd.collect())
    global NUM_USERS
    NUM_USERS = userToRatings_rdd.count()

    #Matrix Id => MovieId
    movieid_indices = ratings_rdd.map(lambda x: x[1][0]).distinct().sortBy(lambda x: x).zipWithIndex().collect()
    # print(movieid_indices)
    #MovieId => Matrix Id
    movie_translation = dict()
    for movie_index in movieid_indices:
        movie_id = movie_index[0]
        index = movie_index[1]
        movie_translation[movie_id] = index
    movie_translation_b = sc.broadcast(movie_translation)
    global NUM_MOVIES
    NUM_MOVIES = len(movie_translation)
    print("Finished Initialization Of RRDs")
    print("There are "+ str(NUM_USERS)+ " users with " + str(NUM_MOVIES) + " movies")
    print("--- %s seconds ---" % (time.time() - start_time))
    R = constructMatrix(userToRatings_rdd,movie_translation_b)
    print("Finished Constructing The Matrix")
    print("--- %s seconds ---" % (time.time() - start_time))
    # print(np.any(np.isnan(R)))
    ms = np.matrix(rand(NUM_USERS, F))
    us = np.matrix(rand(NUM_MOVIES, F))
    Rb = sc.broadcast(R)
    msb = sc.broadcast(ms)
    usb = sc.broadcast(us)
    # # print(ms.shape, "        " , us.shape)
    for i in range(ITERATIONS):
        ms = sc.parallelize(range(NUM_USERS), partitions) \
               .map(lambda x: update(x, usb.value, Rb.value)) \
               .collect()
        # collect() returns a list, so array ends up being
        # a 3-d array, we take the first 2 dims for the matrix
        ms = np.matrix(np.array(ms)[:, :, 0])
        msb = sc.broadcast(ms)

        us = sc.parallelize(range(NUM_MOVIES), partitions) \
               .map(lambda x: update(x, msb.value, Rb.value.T)) \
               .collect()
        us = np.matrix(np.array(us)[:, :, 0])
        usb = sc.broadcast(us)
        error = rmse(R, ms, us)
        print("Iteration: " +str(i) +" has RMSE: "+str(error))
        print("--- %s seconds ---" % (time.time() - start_time))

    print("Finished Calculating The Matrix For Output")
    print("--- %s seconds ---" % (time.time() - start_time))
    output = ms * us.T
    with open('als/matrix.csv','wb') as f:
        for i in range(output.shape[0]):
            f.write(str(output[i,:10]) + "\n")

    
    with open(OUTPUT,'w') as f:
        f.write('user,movie,rating\n')
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                if (not np.isnan(R[i,j])): continue
                userId = i+1
                movieId = movieid_indices[j][0]
                f.write(str(userId) + "," + str(movieId) + "," + str(round(output[i,j],2)) + "\n")
    print("Finished Writing Results To Output File...")
    print("--- %s seconds ---" % (time.time() - start_time))

    

    