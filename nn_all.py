# based on: https://github.com/pushkar/ABAGAIL/blob/master/jython/abalone_test.py

import os
import sys
import time
import math
import csv

sys.path.append("./ABAGAIL-master/ABAGAIL.jar")

from func.nn.backprop import BackPropagationNetworkFactory,BatchBackPropagationTrainer,RPROPUpdateRule
from shared import SumOfSquaresError, DataSet, Instance
from opt.example import NeuralNetworkOptimizationProblem


import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
from func.nn.activation import LinearActivationFunction
import shared.FixedIterationTrainer as FixedIterationTrainer


from func.nn.activation import RELU



def initialize_instances(filename):
    """Read the abalone.txt CSV data into a list of instances."""
    print('Creating instances')
    instances = []

    inFile = open(filename,'rb')
    reader = csv.reader(inFile,delimiter=',')
    
    skipHeader = True
    for row in reader:
        if skipHeader:
            skipHeader = False
            continue
        # print(len(row[1:-1]))
        # print(len([float(value) for value in row[1:-1]]))
        instance = Instance([float(value) for value in row[1:-1]]) #ignore the index and the label
        instance.setLabel(Instance(0 if float(row[-1]) < 15 else 1)) #set the label
        instances.append(instance)
        
    inFile.close()
    
    print('Finished instances')
    
    return instances
    
    
def get_error(data,network,measure):
    error = 0.00
    for j,instance in enumerate(data):
        network.setInputValues(instance.getData())
        network.run()
        
        output = instance.getLabel()
        output_values = network.getOutputValues()
        example = Instance(output_values, Instance(output_values.get(0)))
        error += measure.value(output, example)
    return error


def get_accuracy(data,network):
    correct = 0
    incorrect = 0
    for j,instance in enumerate(data):
        network.setInputValues(instance.getData())
        network.run()
        
        output = instance.getLabel()
        output_values = network.getOutputValues()
        example = Instance(output_values, Instance(output_values.get(0)))
        
        predicted = instance.getLabel().getContinuous()
        actual = network.getOutputValues().get(0)

        if abs(predicted - actual) < 0.1: #three output labels
            correct += 1
        else:
            incorrect += 1
    
    acc = float(correct)/(correct+incorrect)*100.0
    
    return acc
            
            
            
            
def train(oa,network,oaName,measure,train_data,test_data,fout):
    EPOCHS = 601
    times = [0.]
    for epoch in xrange(EPOCHS):
        start = time.clock()
        oa.train()
        times.append(times[-1] + (time.clock()-start))
        
        train_acc = get_accuracy(train_data,network)
        test_acc = get_accuracy(test_data,network)
        
        train_err = get_error(train_data,network,measure)
        test_err = get_error(test_data,network,measure)
        
        row = [oaName,epoch,times[-1],train_acc,test_acc,train_err,test_err]
        print(row)
        fout.writerow(row)
        
        
    
    
def run(alg,oa,fit,classification_network,measure,train_data,test_data,dataSource):
           
    filename = os.path.join('output_data','nn',dataSource+'_nn_'+alg)
    
    columns = ['oa','epoch','time','trainAcc','testAcc','trainErr','testErr']
    outFile = open(filename+'.csv','wb')
    fout = csv.writer(outFile,delimiter=',')
    fout.writerow(columns)
        
        
    train(fit,classification_network,alg,measure,train_data,test_data,fout)
    
    outFile.close()
    
def run_all():
    dataSource = 'wine'
    INPUT_LAYER = 13
    HIDDEN_LAYER = 100
    OUTPUT_LAYER = 1
    
    # dataSource = 'wage'
    # INPUT_LAYER = 106
    # HIDDEN_LAYER = 1000
    # OUTPUT_LAYER = 1

    train_data = initialize_instances('data/balanced_'+dataSource+'_cleaned_train.csv')
    test_data = initialize_instances('data/balanced_'+dataSource+'_cleaned_test.csv')
    factory = BackPropagationNetworkFactory()
    measure = SumOfSquaresError()
    data_set = DataSet(train_data)
    update_rule = RPROPUpdateRule()

    alg = 'backprop'
    classification_network = factory.createClassificationNetwork([INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER],RELU())
    oa = BatchBackPropagationTrainer(data_set,classification_network,measure,update_rule)
    fit = oa
    run(alg,oa,fit,classification_network,measure,train_data,test_data,dataSource)

    alg = 'RHC'
    classification_network = factory.createClassificationNetwork([INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER],RELU())
    nnop = NeuralNetworkOptimizationProblem(data_set,classification_network,measure)
    oa = RandomizedHillClimbing(nnop)
    iters = 1
    fit = FixedIterationTrainer(oa, iters)
    run(alg,oa,fit,classification_network,measure,train_data,test_data,dataSource)
    
    alg = 'SA'
    classification_network = factory.createClassificationNetwork([INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER],RELU())
    nnop = NeuralNetworkOptimizationProblem(data_set,classification_network,measure)
    startTemp = 1E10
    coolingFactor = .8
    oa = SimulatedAnnealing(startTemp,coolingFactor,nnop)
    iters = 1
    fit = FixedIterationTrainer(oa, iters)
    run(alg,oa,fit,classification_network,measure,train_data,test_data,dataSource)
    
    alg = 'GA'
    classification_network = factory.createClassificationNetwork([INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER],RELU())
    nnop = NeuralNetworkOptimizationProblem(data_set,classification_network,measure)
    population = 200
    mates = 50
    mutations = 10
    oa = StandardGeneticAlgorithm(population,mates,mutations,nnop)
    iters = 1
    fit = FixedIterationTrainer(oa, iters)
    run(alg,oa,fit,classification_network,measure,train_data,test_data,dataSource)
    
    
    
    

if __name__ == "__main__":
    run_all()