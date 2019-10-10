import os
import sys
import time
import math

sys.path.append("./ABAGAIL-master/ABAGAIL.jar")


import opt.ga.CrossoverFunction as CrossoverFunction
import opt.ga.SingleCrossOver as SingleCrossOver
import opt.ga.DiscreteChangeOneMutation as DiscreteChangeOneMutation
import opt.ga.SwapMutation as SwapMutation
import opt.DiscreteChangeOneNeighbor as DiscreteChangeOneNeighbor
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.GeneticAlgorithmProblem as GeneticAlgorithmProblem
import opt.ga.MutationFunction as MutationFunction
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.ga.UniformCrossOver as UniformCrossOver
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem

import opt.RandomizedHillClimbing as RandomizedHillClimbing

import opt.SimulatedAnnealing as SimulatedAnnealing

import opt.prob.MIMIC as MIMIC

# abagail tests
import opt.test.FourPeaksTest as FourPeaksTest

# distributions
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import dist.DiscreteDependencyTree as DiscreteDependencyTree


# evaluation functions
import opt.example.ContinuousPeaksEvaluationFunction as ContinuousPeaksEvaluationFunction
import opt.example.FourPeaksEvaluationFunction as FourPeaksEvaluationFunction

# algorithm fitters
import shared.FixedIterationTrainer as FixedIterationTrainer

from array import array

# for cartesian product of inputs, like grid search
from itertools import product
import csv





filename = 'fourpeaks'  

def run_all_2(N=200,T=40,fout=None):
    problem = 'fourpeaks'
    # N=200
    # T=N/10
    maxEpochs = 10**6
    maxTime = 300 #5 minutes
    fill = [2] * N
    ranges = array('i', fill)

    ef = FourPeaksEvaluationFunction(T)
    odd = DiscreteUniformDistribution(ranges)
    nf = DiscreteChangeOneNeighbor(ranges)
    mf = DiscreteChangeOneMutation(ranges)
    # mf = SwapMutation()
    cf = SingleCrossOver()
    df = DiscreteDependencyTree(.1, ranges)
    hcp = GenericHillClimbingProblem(ef, odd, nf)
    gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
    pop = GenericProbabilisticOptimizationProblem(ef, odd, df)
    

    def run_algo(alg,fit,label,difficulty,iters):
        trainTimes = [0.]
        trainTime = []
        scoreChange = [0.]
        stuckCount = 10**3
        prev = 0.
        for epoch in range(0,maxEpochs,1):
        
            st = time.clock()
            fit.train()
            et = time.clock()
            
            trainTimes.append(trainTimes[-1]+(et-st))
            trainTime.append((et-st))
            rollingMean = 10
            avgTime = (math.fsum(trainTime[-rollingMean:]) / float(rollingMean))
        
            score = ef.value(alg.getOptimal())
            
            # trialString = '{}-{}-{}-{}'.format(label,score,epoch,trainTimes[-1])
            trialData = [problem,difficulty,label,score,epoch,trainTimes[-1],avgTime,iters]
            # print(trialData)
            # fout.writerow(trialData)
            # print(trialData)
            print(trialData,max(scoreChange))
            # print(max(scoreChange))
            optimum = (difficulty-1-T) + difficulty
            if score >= optimum: break
            
            scoreChange.append(abs(score-prev))
            prev = score
            scoreChange = scoreChange[-stuckCount:]
            # print(scoreChange)
            if max(scoreChange) == 0: break
            
            if trainTimes[-1] > maxTime: break
            
    
        # print(trialData)
        fout.writerow(trialData)
        
        
    iters = 1000
    rhc = RandomizedHillClimbing(hcp)
    fit = FixedIterationTrainer(rhc, iters)
    run_algo(rhc,fit,'RHC',N,iters)

    iters = 1000
    startTemp = 1E10
    coolingFactor = .95
    sa = SimulatedAnnealing(startTemp, coolingFactor, hcp)
    fit = FixedIterationTrainer(sa, iters)
    run_algo(sa,fit,'SA',N,iters)

    iters = 10
    population = 300
    mates = 100
    mutations = 50
    ga = StandardGeneticAlgorithm(population, mates, mutations, gap)
    fit = FixedIterationTrainer(ga, iters)
    run_algo(ga,fit,'GA',N,iters)
    
    iters = 10
    samples = 200
    keep = 20
    mimic = MIMIC(samples, keep, pop)
    fit = FixedIterationTrainer(mimic, iters)
    run_algo(mimic,fit,'MIMIC',N,iters)
    

def vary_difficulty():
    columns = ['problem','difficulty','label','score','epoch','time','avgTrainTime','iterations']
    # outFile = open(filename+'_all.csv','wb')
    outFile = open(os.path.join('output_data',filename+'_difficulty_all.csv'),'wb')
    fout = csv.writer(outFile,delimiter=',')
    fout.writerow(columns)
    
    # maxN = 110
    problem_sizes = [10,20,50,75,100,125,150]
    # problem_sizes = [10,50,100,150]
    for i in problem_sizes:
        T = i/10
        run_all_2(i,T,fout)
        
    outFile.close()


def run():
    print('Hello')
    vary_difficulty()
    # run_all()
    # fp_test()


if __name__ == '__main__':
    run()