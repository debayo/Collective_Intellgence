# coding: utf-8
'''
Created on Jun 15, 2013

@author: Dell390
'''
import numpy as np
import collections
import time
# A dictionary of movie critics and their ratings of a small
# set of movies
critics = {'Lisa Rose': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.5,
 'Just My Luck': 3.0, 'Superman Returns': 3.5, 'You, Me and Dupree': 2.5,
 'The Night Listener': 3.0},
'Gene Seymour': {'Lady in the Water': 3.0, 'Snakes on a Plane': 3.5,
 'Just My Luck': 1.5, 'Superman Returns': 5.0, 'The Night Listener': 3.0,
 'You, Me and Dupree': 3.5},
'Michael Phillips': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.0,
 'Superman Returns': 3.5, 'The Night Listener': 4.0},
'Claudia Puig': {'Snakes on a Plane': 3.5, 'Just My Luck': 3.0,
 'The Night Listener': 4.5, 'Superman Returns': 4.0,
 'You, Me and Dupree': 2.5},
'Mick LaSalle': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
 'Just My Luck': 2.0, 'Superman Returns': 3.0, 'The Night Listener': 3.0,
 'You, Me and Dupree': 2.0},
'Jack Matthews': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
 'The Night Listener': 3.0, 'Superman Returns': 5.0, 'You, Me and Dupree': 3.5},
'Toby': {'Snakes on a Plane':4.5, 'You, Me and Dupree':1.0, 'Superman Returns':4.0}}

from math import sqrt

# 返回person1 person2基于欧几里得距离的相似度评价
def simDistance(prefs, person1, person2):
  '''返回person1 person2基于欧几里得距离的相似度评价
  '''
  commonItems = [item for item in prefs[person1] if item in prefs[person2]]
  if len(commonItems) == 0:
    return 0
  else:
    sumOfSquares = sum([pow((prefs[person1][item] - prefs[person2][item]), 2) for item in commonItems])
    return 1 / (1 + sqrt(sumOfSquares))  # 防止分母为1

# 返回p1和p2的皮尔逊相关系数
def simPearson(prefs, p1, p2):
  '''返回p1和p2的皮尔逊相关系数
     基于X和Y向量标准化的后的余弦相似度
  '''
  commonItems = [item for item in prefs[p1] if item in prefs[p2]]
  Nc = len(commonItems)
  if Nc == 0:
    return 0
  
  X = np.array([prefs[p1][i] for i in commonItems])
  Y = np.array([prefs[p2][i] for i in commonItems])
  X_avg = sum(X) / Nc
  Y_avg = sum(Y) / Nc
  X_var = sqrt(Nc * sum([pow(item, 2) for item in X]) - sum(X)) / Nc
  Y_var = sqrt(Nc * sum([pow(item, 2) for item in Y]) - sum(Y)) / Nc
  X = (X - X_avg) / X_var
  Y = (Y - Y_avg) / Y_var
  return np.dot(X, Y) / sqrt(sum(X * X) * sum(Y * Y))
    
    
# 返回p1和p2的皮尔逊相关系数
def sim_pearson_my(prefs, person1, person2):
  '''返回p1和p2的皮尔逊相关系数
  '''
  commonItems = [item for item in prefs[person1] if item in prefs[person2]]
  Nc = len(commonItems)
  if Nc == 0:
    return 0
  
  sum_xy = sum([prefs[person1][i] * prefs[person2][i] for i in commonItems])
  sum_x = sum([prefs[person1][i] for i in commonItems])
  sum_y = sum([prefs[person2][i] for i in commonItems])
  sum_x_square = sum([pow(prefs[person1][i], 2) for i in commonItems])
  sum_y_square = sum([pow(prefs[person2][i], 2) for i in commonItems])
  
  num = Nc * sum_xy - sum_x * sum_y;
  den = sqrt(Nc * sum_x_square - pow(sum_x, 2)) * sqrt(Nc * sum_y_square - pow(sum_y, 2))
  if den == 0: return 0;
  sim = num / den
  return sim
           
def topMatches(prefs, person, n=5, similarity=simPearson):
  '''返回最相似前n个实例
  '''
  scores = [(similarity(prefs, person, other), other) for other in prefs if other != person]
  scores.sort(reverse=True)
  return scores[0:n]


# 将物品与人员对调
def transformPrefs(prefs):
  '''将物品与人员对调
  '''
  result = {}  
  for person in prefs:
    for item in prefs[person]:
      result.setdefault(item, {})
      result[item][person] = prefs[person][item]
  return result

# 建立字典，以给出与这些物品最相似的所有其他物品集合
def calculateSimilarItems(prefs, n=10):
  '''建立字典，以给出与这些物品最相似的所有其他物品集合
  ''' 
  result = {}
  c = 0
  itemPrefs = transformPrefs(prefs)
  
  for item in itemPrefs:
    c += 1
    if c % 100 == 0: print"%d / %d", (c, len(itemPrefs))
    scores = topMatches(itemPrefs, item, n=n, similarity=simDistance)             
    result[item] = scores
  return result


def getRecommendations(prefs, person, similarity=sim_pearson_my):
  totals = {}
  simSums = {}
  
  for other in prefs:
    if(other != person):
      sim = similarity(prefs, person, other)
      
      if sim <= 0:
        continue
      
      for item in prefs[other]:
        if item not in prefs[person] or prefs[person][item] == 0:
          totals.setdefault(item, 0)
          totals[item] += prefs[other][item] * sim
          
          simSums.setdefault(item, 0)
          simSums[item] += sim
  rankings = [(total / simSums[item], item) for item, total in totals.items()]
  rankings.sort(key=None, reverse=True)
  return rankings

def getRecommendationsMy(prefs, person, similarity=sim_pearson_my):
  totals = collections.defaultdict(float)
  simSums = collections.defaultdict(float)
  # 按向量求解，而不是一个一个求解
  for other in prefs:
    if other != person:
      sim = similarity(prefs, person, other)
      if sim <= 0: continue  # 两个user打分的Item项目没有交集
      
      for item in prefs[other]:
        if item not in prefs[person]:
          totals[item] += sim * prefs[other][item]
          simSums[item] += sim
  
  rankings = [(total / simSums[item], item) for item, total in totals.items()]
  rankings.sort(reverse=True)
  return rankings


def getRecommendationsItems(prefs, itemSim, user):
  userRatings=prefs[user]
  totals = collections.defaultdict(float)
  simSums = collections.defaultdict(float)
  #
  for (item,rating) in userRatings.items():
    for (sim, item2) in itemSim[item]:
      if item2 in userRatings: continue
      totals[item2] += sim * rating
      simSums[item2] += sim
  
  rankings = [(total / simSums[item], item) for item, total in totals.items()]
  rankings.sort(reverse=True)
  return rankings
# 使用movielens数据集
def loadMovieLens(path='E:/DataSet/ml-1m/ml-1m/'):
  '''加载movielens数据集
  '''
  #获得影片标题
  movies = {}
  for line in open(path+'movies.dat'):
    (id,item) = line.split('::')[0:2]
    movies[id] = item
  #加载数据
  prefs = {}
  for line in open(path+'ratings.dat'):
    (userID,movieID,rating,timestamp) = line.split('::')
    prefs.setdefault(userID,{})
    prefs[userID][movies[movieID]] = float(rating)
  return prefs  
      
# similarity test
print 'similarity test'
print(simDistance(critics, 'Lisa Rose', 'Gene Seymour'))
print(simPearson(critics, 'Lisa Rose', 'Gene Seymour'))
print(sim_pearson_my(critics, 'Lisa Rose', 'Gene Seymour'))

# topMatches
print 'topMatches'
scores = topMatches(critics, 'Toby', n=3)
print(scores)


# getRecommendation
print 'getRecommendation'
print(getRecommendations(critics, 'Toby'))
print(getRecommendations(critics, 'Toby', similarity=simDistance))

print(getRecommendationsMy(critics, 'Toby'))
print(getRecommendationsMy(critics, 'Toby', similarity=simDistance))
# Reverse Person and movies
# get Recommendation
print 'Reverse Person and movies '
movies = transformPrefs(critics)
scoress = topMatches(movies, 'Superman Returns')
print(scoress)
print 'get Recommendation '
print(getRecommendations(movies, 'Just My Luck'))

# calculateSimilarItems
print 'calculateSimilarItems'
itemSim = calculateSimilarItems(critics, n=10)
print itemSim
# getRecommendationsItems
print 'getRecommendationsItems'
print getRecommendationsItems(critics, itemSim, 'Toby')

print 'loadMovieLens'
start = time.clock()
prefs = loadMovieLens()
print 'getRecommendationsMy(prefs,\'87\')[0:30]', getRecommendationsMy(prefs,'87')[0:30]
end = time.clock()
print 'loadMovieLens and getRecommendationsMy cost %d seconds' % (end-start)
