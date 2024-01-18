#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 18:02:23 2024

@author: ubuntu
"""
import numpy as np

def evolve(indiv, policy, mutation, ngens=10, podium=3, popsize=100, keep_dad=True):
    population=[indiv]*popsize
    scores=[0]*popsize
    print("\nbase score: {0}".format(policy(indiv)))
    
    if keep_dad:
        for t in range(ngens):
            for i in range(1,popsize):
                population[i] = mutation(population[i])
                scores[i] = policy(population[i])
            population[0] = indiv
            scores[0] = policy(indiv)
            meanscore = np.mean(scores)
            maxscore = np.max(scores)
            print("gen {0} avg. score: {1}\tmax: {2}".format(t+1, meanscore,maxscore))
            
            winners_ids = np.argsort(scores)[::-1][:podium]
            population = list(np.array(population)[list(np.random.choice(winners_ids, size=popsize))])
    else:
        for t in range(ngens):
            for i in range(popsize):
                population[i] = mutation(population[i])
                scores[i] = policy(population[i])
            meanscore = np.mean(scores)
            maxscore = np.max(scores)
            print("gen {0} avg. score: {1}\tmax: {2}".format(t+1, meanscore,maxscore))
            winners_ids = np.argsort(scores)[::-1][:podium]
            population = list(np.array(population)[list(np.random.choice(winners_ids, size=popsize))])
    return list(np.array(population)[list(winners_ids)])
    
