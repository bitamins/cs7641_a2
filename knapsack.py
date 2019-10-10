import sys
import os
import time

import java.io.FileReader as FileReader
import java.io.File as File
import java.lang.String as String
import java.lang.StringBuffer as StringBuffer
import java.lang.Boolean as Boolean
import java.util.Random as Random

sys.path.append("./ABAGAIL-master/ABAGAIL.jar")

import dist.DiscreteDependencyTree as DiscreteDependencyTree
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import dist.Distribution as Distribution
import opt.DiscreteChangeOneNeighbor as DiscreteChangeOneNeighbor
import opt.EvaluationFunction as EvaluationFunction
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.HillClimbingProblem as HillClimbingProblem
import opt.NeighborFunction as NeighborFunction
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.example.FourPeaksEvaluationFunction as FourPeaksEvaluationFunction
import opt.ga.CrossoverFunction as CrossoverFunction
import opt.ga.SingleCrossOver as SingleCrossOver
import opt.ga.DiscreteChangeOneMutation as DiscreteChangeOneMutation
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.GeneticAlgorithmProblem as GeneticAlgorithmProblem
import opt.ga.MutationFunction as MutationFunction
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.ga.UniformCrossOver as UniformCrossOver
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC as MIMIC
import opt.prob.ProbabilisticOptimizationProblem as ProbabilisticOptimizationProblem
import shared.FixedIterationTrainer as FixedIterationTrainer

import opt.example.KnapsackEvaluationFunction as KnapsackEvaluationFunction

from array import array
# for cartesian product of inputs, like grid search
from itertools import product
import csv
import math


"""
Commandline parameter(s):
   none
"""
# fill a knapsack with the certain volumed items to maximize the value of items in the knapsack

filename = 'knapsack'

def run_all():
    problem = 'knapsack'
    # Random number generator */
    random = Random()
    # The number of items
    NUM_ITEMS = 40
    # The number of copies each
    COPIES_EACH = 4
    # The maximum weight for a single element
    MAX_WEIGHT = 50
    # The maximum volume for a single element
    MAX_VOLUME = 50
    # The volume of the knapsack 
    KNAPSACK_VOLUME = MAX_VOLUME * NUM_ITEMS * COPIES_EACH * .4

    # create copies
    fill = [COPIES_EACH] * NUM_ITEMS
    copies = array('i', fill)

    # create weights and volumes
    fill = [0] * NUM_ITEMS
    weights = array('d', fill)
    volumes = array('d', fill)
    for i in range(0, NUM_ITEMS):
        weights[i] = random.nextDouble() * MAX_WEIGHT
        volumes[i] = random.nextDouble() * MAX_VOLUME


    # create range
    fill = [COPIES_EACH + 1] * NUM_ITEMS
    ranges = array('i', fill)

    ef = KnapsackEvaluationFunction(weights, volumes, KNAPSACK_VOLUME, copies)
    odd = DiscreteUniformDistribution(ranges)
    nf = DiscreteChangeOneNeighbor(ranges)
    mf = DiscreteChangeOneMutation(ranges)
    cf = UniformCrossOver()
    df = DiscreteDependencyTree(.1, ranges)
    hcp = GenericHillClimbingProblem(ef, odd, nf)
    gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
    pop = GenericProbabilisticOptimizationProblem(ef, odd, df)

    maxEpochs = 400

    columns = ['problem','label','score','epoch','time','avgTrainTime','iterations']
    outFile = open(filename+'_all.csv','wb')
    fout = csv.writer(outFile,delimiter=',')
    fout.writerow(columns)


    def run_algo(alg,fit,label,iters):
        print(alg)
        trainTimes = [0.]
        trainTime = []
        scores = [0]
        deltaScores = []
        for epoch in range(0,maxEpochs,1):
        
            st = time.clock()
            fit.train()
            et = time.clock()
            
            trainTimes.append(trainTimes[-1]+(et-st))
            trainTime.append((et-st))
            rollingMean = 10
            avgTime = (math.fsum(trainTime[-rollingMean:]) / float(rollingMean))
      
            score = ef.value(alg.getOptimal())
            scores.append(score)
            deltaScores.append(math.fabs(scores[-2] - scores[-1]))
            
            # trialString = '{}-{}-{}-{}'.format(label,score,epoch,trainTimes[-1])
            trialData = [problem,label,score,epoch,trainTimes[-1],avgTime,iters]
            print(trialData)
            fout.writerow(trialData)
            
            
    iters = 10
    rhc = RandomizedHillClimbing(hcp)
    fit = FixedIterationTrainer(rhc, iters)
    run_algo(rhc,fit,'RHC',10)


    startTemp = 1E11
    coolingFactor = .95
    sa = SimulatedAnnealing(startTemp, coolingFactor, hcp)
    fit = FixedIterationTrainer(sa, iters)
    run_algo(sa,fit,'HCP',10)

    population = 300
    mates = 100
    mutations = 50
    ga = StandardGeneticAlgorithm(population, mates, mutations, gap)
    fit = FixedIterationTrainer(ga, iters)
    run_algo(ga,fit,'GA',10)
    
    
    samples = 200
    keep = 20
    mimic = MIMIC(samples, keep, pop)
    fit = FixedIterationTrainer(mimic, iters)
    run_algo(mimic,fit,'MIMIC',10)
    
    outFile.close()
    
    
def run_all_2(N=40,fout=None):
    maxEpochs = 10**5
    maxTime = 300 #5 minutes
    problem = 'knapsack'
    # Random number generator */
    random = Random()
    # The number of items
    NUM_ITEMS = N
    # The number of copies each
    COPIES_EACH = 4
    # The maximum weight for a single element
    MAX_WEIGHT = 50
    # The maximum volume for a single element
    MAX_VOLUME = 50
    # The volume of the knapsack 
    KNAPSACK_VOLUME = MAX_VOLUME * NUM_ITEMS * COPIES_EACH * .4

    # create copies
    fill = [COPIES_EACH] * NUM_ITEMS
    copies = array('i', fill)

    # create weights and volumes
    fill = [0] * NUM_ITEMS
    weights = array('d', fill)
    volumes = array('d', fill)
    for i in range(0, NUM_ITEMS):
        weights[i] = random.nextDouble() * MAX_WEIGHT
        volumes[i] = random.nextDouble() * MAX_VOLUME


    # create range
    fill = [COPIES_EACH + 1] * NUM_ITEMS
    ranges = array('i', fill)

    ef = KnapsackEvaluationFunction(weights, volumes, KNAPSACK_VOLUME, copies)
    odd = DiscreteUniformDistribution(ranges)
    nf = DiscreteChangeOneNeighbor(ranges)
    mf = DiscreteChangeOneMutation(ranges)
    cf = UniformCrossOver()
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
            # optimum = (difficulty-1-T) + difficulty
            # if score >= optimum: break
            
            scoreChange.append(abs(score-prev))
            prev = score
            scoreChange = scoreChange[-stuckCount:]
            # print(scoreChange)
            if max(scoreChange) < 1.0: break
            
            if trainTimes[-1] > maxTime: break
            
    
        # print(trialData)
        fout.writerow(trialData)
        
        
    iters = 1000
    rhc = RandomizedHillClimbing(hcp)
    fit = FixedIterationTrainer(rhc, iters)
    run_algo(rhc,fit,'RHC',N,iters)

    iters = 1000
    startTemp = 1E10
    coolingFactor = .99
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
    # problem_sizes = [10,50,100,150,200]
    problem_sizes = [10,50,100,150]
    for i in problem_sizes:
        run_all_2(i,fout)
        
    outFile.close()
        
        
def run():
    print('Hello')
    # run_all()
    vary_difficulty()


if __name__ == '__main__':
    run()