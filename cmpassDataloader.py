
from cmapssdata import CMAPSSDataset
import numpy as np
class dataLoader():
    def __init__(self, bs = 10, sl = 50):
        self.batchSize = bs
        self.sequence_length = sl
        cmpass = CMAPSSDataset(fd_number='1', batch_size=bs, sequence_length=sl)
        trainData = cmpass.get_train_data()
        self.trainDataFeature = cmpass.get_feature_slice(trainData)
        self.trainDataLabel = cmpass.get_label_slice(trainData)
        self.trainEngineID = cmpass.get_engine_id(trainData)
        index = np.random.permutation(self.trainDataFeature.shape[0])
        self.trainDataFeature = self.trainDataFeature[index,:,:]
        self.trainDataLabel = self.trainDataLabel[index]
        self.trainEngineID = self.trainEngineID[index]
        testData = cmpass.get_test_data()
        self.testDataFeature = cmpass.get_feature_slice(testData)
        self.testDataLabel = cmpass.get_label_slice(testData)
        self.testEngineID = cmpass.get_engine_id(testData)
        self.batches = self.trainDataFeature.shape[0]//self.batchSize
        self.batchpoint = 0


    def nextBatch(self):
        if self.batchpoint >= self.batches - 1:
            self.batchpoint = 0
            return self.trainDataFeature[(self.batches-1)* self.batchSize : self.batches*self.batchSize,:,:],\
                   self.trainDataLabel[(self.batches-1) * self.batchSize : self.batches*self.batchSize]


        else:
            self.batchpoint = self.batchpoint + 1
            return self.trainDataFeature[(self.batchpoint-1) * self.batchSize:self.batchpoint*self.batchSize,:,:],\
                   self.trainDataLabel[(self.batchpoint-1) * self.batchSize:self.batchpoint*self.batchSize]

if __name__ == '__main__':
    d = dataLoader(10,50)
    z , v= d.nextBatch()
    co = 1






