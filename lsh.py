from pyspark import SparkContext
import sys,os
import numpy as np
import time
#############Util Functions#############
#Utility functions to create signatures
def hash(i,movieId):
    return (3 * movieId + 11*i ) % 100 + 1

#Construct baskets rdd from file
def getBasketsRdd():
    file = sc.textFile(INPUT, minPartitions=NUM_PARTITIONS)
    ratings = file.map(splitRow).filter(lambda x: x[0] != 'userId' and x[1] != 'movieId').map(lambda x: (int(x[0]),(int(x[1]),float(x[2]))))
    return ratings

#Split a row in text file, util function
#userId,movieId,rating
def splitRow(row):
    words = row.split(",")
    return (words[0],words[1],words[2])
    
#######################################


########### Construct Signatures#############
#Input: (UserId,(movieId,Rating))
#(userId,[50 min hashes])
def constructUserMinHash(ratings_rdd):
    return ratings_rdd.groupByKey().map(minHashCombiner).reduceByKey(minHashReducer).sortBy(lambda x: x[0])

#Input: User => Signatures of each has
#Output: Signature index => [signatures]
def constructSignatures(userMinhash_rdd):
    signatures_rdd = userMinhash_rdd.flatMap(lambda x : getAllPairs(x)).groupByKey().map(lambda x: list(sorted(x[1])))
    return signatures_rdd

# #userIdToMinHash => (userId,[50 min hashes])
# #return => [(rowNum,(userId,minHashValue))]
def getAllPairs(userIdTominHashRow):
    userId = userIdTominHashRow[0]
    lst = []
    for tup in enumerate(userIdTominHashRow[1]):
        index,hash_value= tup
        lst.append((index,(userId,hash_value)))

    return lst
#row => (userId,[movieId1, movieId2])
#return => (userId => [50 min hashes])
def minHashCombiner(row):
    userId = row[0]
    hashes = []
    for i,tup in enumerate(row[1]):
        for j in range(50):
            if i == 0:
                hashes.append(hash(j+1,tup[0]))
            else:
                hashes[j] = min(hash(j+1,tup[0]),hashes[j])
    return (userId,hashes)


###############Construct User => Movie => Rating##################### 
def generateMatrixRow(user_ratings,movie_translation_b):
    userId = user_ratings[0]
    ratings = user_ratings[1]
    return_row = np.empty((1,NUM_MOVIES))*(np.nan)
    for (movieId,rating) in ratings:
        index = movie_translation_b.value[movieId]
        return_row[0][index] = rating
    return (userId,return_row)

def minHashReducer(hash1, hash2):
    lst = []
    for i in range(len(hash1)):
        lst.append(min(hash1[i],hash2[i]))
    return lst

def constructMatrix(userToRatings_rdd,movie_translation_b):
    temp = userToRatings_rdd.map(lambda x:generateMatrixRow(x,movie_translation_b))\
        .sortBy(lambda x: x[0])
    matrix = temp.map(lambda x: x[1]).reduce(lambda x,y: merge(x,y))
    return matrix

def merge(row1,row2):
    return np.concatenate((row1,row2),axis = 0)
#######################################

###########LSH#############
def constructAllCandidateSets(matrix_rdd):
    candidates_all = matrix_rdd.mapPartitions(generateCandidateSets)
    return candidates_all

def generateCandidateSets(partition):
    dictionary = {}
    candidates = []
    for lst in partition:
        for tup in lst:
            signatureRowIdx = tup[0]

            if signatureRowIdx not in dictionary:
                dictionary[signatureRowIdx] = []
            dictionary[signatureRowIdx].append(tup[1])

    keys = dictionary.keys()
    for i in range(len(keys)):
        for j in range(i+1,len(keys)):
            key1 = keys[i]
            key2 = keys[j]
            list1 = dictionary[key1]
            list2 = dictionary[key2]
            assert(len(list1) == 50/NUM_PARTITIONS)
            assert(len(list2) == 50/NUM_PARTITIONS)
            identical = True
            for k in range(len(list1)):
                if list1[k] != list2[k]:
                    identical = False
                    break
            if identical:
                if key1 < key2:
                    candidates.append((key1,key2))
                else:
                    candidates.append((key2,key1))

    return candidates



###########Find The Three Most Similar Users For Each User#################
def groupRatings(similarity_pair):
    user1 = similarity_pair[0]
    user2 = similarity_pair[1]
    similarity = similarity_pair[2]
    return [(user1,(user2,similarity)),(user2,(user1,similarity))]

def getThreeMostSimilar(user_jaccard):
    temp = sorted(user_jaccard[1],reverse = True,key = lambda x: x[1])
    temp = temp[:3]
    return (user_jaccard[0],temp)

def calculate(pair,user_movie_rating_b):
    user1 = pair[0]
    user2 = pair[1]
    user1Ratings = user_movie_rating_b.value[user1-1]
    user2Ratings = user_movie_rating_b.value[user2-1]
    user1RatedMovies = set(np.ravel(np.argwhere(~np.isnan(user1Ratings))))
    user2RatedMovies = set(np.ravel(np.argwhere(~np.isnan(user2Ratings))))
    bothRatedMovies = user1RatedMovies.intersection(user2RatedMovies)
    allRatedMovies = user1RatedMovies.union(user2RatedMovies)
    jaccard = float(len(bothRatedMovies)) / float(len(allRatedMovies))
    return (pair[0],pair[1],jaccard)

def calculateSimilarity(candidates,userMinHash_b):
    result = candidates.map(lambda x: calculate(x,userMinHash_b)).flatMap(lambda x: groupRatings(x))\
            .groupByKey().map(lambda x: getThreeMostSimilar(x))
    return result

################################################################################

##############Write the Average Ratings for Missing Values##############################
def getAllAverageRatings(similarities,movieid_indices,user_movie_rating,file):
    for i,user_similar_users in enumerate(similarities):
        userId = user_similar_users[0]
        similar_users = user_similar_users[1]
        userRow = user_movie_rating[userId-1,:]
        for j in range(userRow.shape[0]):
            if not np.isnan(userRow[j]):continue
            count = 0
            summ = 0.0
            for user in similar_users:
                rating = user_movie_rating[user-1,j]
                if np.isnan(rating): continue
                count += 1
                summ += rating
            if count ==0 :continue
            res = round(float(summ) / float(count),3)
            movieId = movieid_indices[j][0]
            f.write(str(userId) + "," + str(movieId) + "," + str(res) + "\n")
            
# def getAllAverageRatings(similarities,movieid_indices,user_movie_rating,file):
    # for j,user_similar_users in enumerate(similarities):
    #     userId = user_similar_users[0]
    #     similar_users_indices = [x-1 for x in user_similar_users[1]]
        # # print(similar_users_indices.shape)
        # user_row = user_movie_rating[userId-1]
        # tofill = np.ravel(np.argwhere(np.isnan(user_row)))
        # # print(tofill.shape,similar_users_indices.shape)
        # similar_users_rows = user_movie_rating[similar_users_indices,:]
        # similar_users_rows = user_movie_rating[:,tofill]
        # # print(similar_users_rows.shape)
        # avg_ratings =np.nanmean(similar_users_rows,axis=0)
        # avg_effective_ratings_index = np.ravel(np.argwhere(~np.isnan(avg_ratings)))
        # if j == 0:
        #     print(userId)
        #     print(avg_effective_ratings_index)
        # for i in range(len(avg_effective_ratings_index)):
        #     fillIdx = movieid_indices[tofill[i]][0]
        #     file.write(str(userId) + ","+str(fillIdx) + "," + str(avg_ratings[i]) + "\n")
    

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('You also need to pass in the input and output file')
        print('spark-submit lsh.py ratings.csv lsh/predictions.csv')
        sys.exit(1)
    if 'als' not in os.listdir('.'):
        os.mkdir('als')
    global start_time
    start_time = time.time()
    #INPUT file for the program
    global INPUT
    INPUT = sys.argv[1]
    global OUTPUT
    OUTPUT = sys.argv[2]
    global NUM_PARTITIONS 
    #**** The Number of Bands ****#
    NUM_PARTITIONS = 10
    sc = SparkContext(appName = "Homework 3")
    #***Getting The Ratings Rdd => (userId,(movieId,rating))
    ratings_rdd = getBasketsRdd().cache()

    print("Finished Reading Constructing Ratings RDD")
    print("--- %s seconds ---" % (time.time() - start_time))
    #***Matrix row index to Movie Indices in user_movie_rating matrix***
    movieid_indices = ratings_rdd.map(lambda x: x[1][0]).distinct().sortBy(lambda x: x).zipWithIndex().collect()
    #***Dictionary of movie Id to row indices in user_movie_rating matrix***
    movie_translation = dict()
    for movie_index in movieid_indices:
        movie_id = movie_index[0]
        index = movie_index[1]
        movie_translation[movie_id] = index

    #***Number of Movies in Total
    global NUM_MOVIES
    NUM_MOVIES = len(movie_translation)
    movie_translation_b = sc.broadcast(movie_translation)
    movieid_indices_b = sc.broadcast(movieid_indices)
    print("Finished Creating Movie and Index Mapping")
    print("--- %s seconds ---" % (time.time() - start_time))
    #Construct The User => MovieId => Ratings Pair
    userToRatings_rdd = ratings_rdd.groupByKey()
    user_movie_rating = constructMatrix(userToRatings_rdd,movie_translation_b)
    user_movie_rating_b = sc.broadcast(user_movie_rating)
    print("Finished Creating User Movie Ratings Matrix")
    print("--- %s seconds ---" % (time.time() - start_time))
    #Construct signatures, Without LSH, There would be 185745 Comparisons
    userMinHash_rdd = constructUserMinHash(ratings_rdd).cache()
    userMinHash = userMinHash_rdd.collect()
    global NUM_USERS
    NUM_USERS = len(userMinHash)
    userMinHash_b = sc.broadcast(userMinHash)
    print("Finished Constructing User Min Hash Table. There are " + str(NUM_MOVIES) + " movies and " + str(NUM_USERS) + " users")
    print("--- %s seconds ---" % (time.time() - start_time))
    # print(userMinHash)
    #####################

    #Row Number To Users Signature
    signatures_rdd = constructSignatures(userMinHash_rdd)
    # print(signatures_rdd.take(1))
    candidates_rdd = constructAllCandidateSets(signatures_rdd).distinct()
    with open('lsh/candidates.csv','wb') as f:
        for cand in candidates_rdd.collect():
            f.write(str(cand) + "\n")
    print("Candidates Selection Complete, Reduced Pairs From " + str(NUM_USERS*(NUM_USERS-1)) + " to " + str(candidates_rdd.count()))
    print("--- %s seconds ---" % (time.time() - start_time))
    #***
    similarities = calculateSimilarity(candidates_rdd,user_movie_rating_b).map(lambda x: (x[0],[usr[0] for usr in x[1]])).sortBy(lambda x: x[0]).collect()
    with open('lsh/similarities.csv','wb') as f:
        for similarity in similarities:
            f.write(str(similarity) + "\n")
    # print(getAllAverageRatings(similarities,userMinHash,user_movie_rating))

    # print("Finished Selecting Top Three Most Similar Users for Each User")
    # print("--- %s seconds ---" % (time.time() - start_time))
    # result = getAllAverageRatings(similarities,movieid_indices=,user_movie_rating)
    # print("Finished The Overall Result")
    # print("--- %s seconds ---" % (time.time() - start_time))
    with open(OUTPUT,'wb') as f:
        f.write('user,movie,rating\n')
        getAllAverageRatings(similarities,movieid_indices,user_movie_rating,f)
    print("Finished Getting Averages And Writing Results To File")
    print("--- %s seconds ---" % (time.time() - start_time))