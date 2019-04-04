import pdb 
import numpy as np
import pandas as pd
import os
import torch

pd.set_option('display.max_columns',None)
def convert_timestr2int(ts):
    a=ts.str.split(':')
    return a.map(lambda x:float(x[0])*3600+float(x[1])*60+float(x[2]))
def modification(t1,t2):
    if t2<t1:
        t2=t2+24*3600
    return t2
def convert_df(df):
    # convert the first check in
    # for i in range(df.shape[0]):
    #     temp_1 = df.iloc[i, 3]
    #     temp_2 = df.iloc[i, 6]
    #     temp_3 = df.iloc[i, 9]
    #
    #     df.iloc[i, 3] = convert_timestr2int(temp_1)
    #     df.iloc[i, 6] = modification(df.iloc[i, 3], convert_timestr2int(temp_2))
    #     df.iloc[i, 9] = modification(df.iloc[i, 6], convert_timestr2int(temp_3))
    #     if df.shape[1]==13:
    #         temp_4 = df.iloc[i, 12]
    #         df.iloc[i, 12] = modification(df.iloc[i, 9], convert_timestr2int(temp_4))////
    s=df.shape[1]
    df['f']=convert_timestr2int(df[3])
    df['s'] = convert_timestr2int(df[6])
    df['t'] = convert_timestr2int(df[9])

    df[3] = df['f']/24/3600
    df[6] = df.apply(lambda x: modification(x[ 3], x['s']),axis=1)
    df[9] = df.apply(lambda x: modification(x[6], x['t']),axis=1)
    # if s == 13:
    df['ft'] = convert_timestr2int(df[12])
    df[12] = df.apply(lambda x: modification(x[9],x['ft']),axis=1)
    df[1]=df[1]/2915
    df[2]=df[2]/1982
    df[4] = df[4] / 2915
    df[7] = df[7] / 2915
    df[10] = df[10] / 2915
    df[5] = df[5] / 1982
    df[8] = df[8] / 1982
    df[11] = df[11] / 1982


def data():
    # print(os.getcwd())
    train=pd.read_csv('./social-checkin-prediction/train.csv',header=None)
    test=pd.read_csv('./social-checkin-prediction/test.csv',header=None)
    validate=pd.read_csv('./social-checkin-prediction/validation.csv',header=None)

    # print(train.head(10))
    # print()
    # print(test.head(10))
    # print()
    # print(validate.head(10))
    # print()
    # trainx=train.values[:,1:10]
    # trainy=train.values[:,10:]
    # validatex=validate.values[:,1:10]
    # validatey=validate.values[:,10:]
    # testx=test.values[:,1:10]
    # print (trainx[0,:])
    # print (trainy[0])
    # print (validatex[0,:])
    # print (validatey[0])

    convert_df(train)
    convert_df(validate)
    convert_df(test)
    # print(train.head(10))
    # print()
    # print(test.head(10))
    # print()
    # print(validate.head(10))
    # print()
    trainx=train.values[:,1:10]
    trainy=train.values[:,10:13]
    validatex=validate.values[:,1:10]
    validatey=validate.values[:,10:13]
    testx=test.values[:,1:10]
    testy=test.values[:,10:13]
    return torch.from_numpy(trainx), torch.from_numpy(trainy), torch.from_numpy(validatex), torch.from_numpy(validatey), torch.from_numpy(testx), torch.from_numpy(testy)


class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.structure=torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.BatchNorm1d(H),
        torch.nn.Dropout(0.25),
        torch.nn.Linear(H, H//4),
        torch.nn.ReLU(),
        torch.nn.BatchNorm1d(H//4),
        torch.nn.Dropout(0.25),
        torch.nn.Linear(H//4,D_out))
    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        return self.structure(x)


if __name__=='__main__':
    trainx,trainy,validatex,validatey,testx,testy = data()
    trainx[:,0:1] = (trainx[:,0:1] - torch.min(trainx[:,0:1])) / (torch.max(trainx[:,0:1]) - torch.min(trainx[:,0:1]))
    trainx[:,1:2] = (trainx[:,1:2] - torch.min(trainx[:,1:2])) / (torch.max(trainx[:,1:2]) - torch.min(trainx[:,1:2]))
    trainx[:,2:3] = (trainx[:,2:3] - torch.min(trainx[:,2:3])) / (torch.max(trainx[:,2:3]) - torch.min(trainx[:,2:3]))

    trainy[:,0:1] = (trainy[:,0:1] - torch.min(trainy[:,0:1])) / (torch.max(trainy[:,0:1]) - torch.min(trainy[:,0:1]))
    trainy[:,1:2] = (trainy[:,1:2] - torch.min(trainy[:,1:2])) / (torch.max(trainy[:,1:2]) - torch.min(trainy[:,1:2]))
    trainy[:,2:3] = (trainy[:,2:3] - torch.min(trainy[:,2:3])) / (torch.max(trainy[:,2:3]) - torch.min(trainy[:,2:3]))

    validatex[:,0:1] = (validatex[:,0:1] - torch.min(validatex[:,0:1])) / (torch.max(validatex[:,0:1]) - torch.min(validatex[:,0:1]))
    validatex[:,1:2] = (validatex[:,1:2] - torch.min(validatex[:,1:2])) / (torch.max(validatex[:,1:2]) - torch.min(validatex[:,1:2]))
    validatex[:,2:3] = (validatex[:,2:3] - torch.min(validatex[:,2:3])) / (torch.max(validatex[:,2:3]) - torch.min(validatex[:,2:3]))

    validatey[:,0:1] = (validatey[:,0:1] - torch.min(validatey[:,0:1])) / (torch.max(validatey[:,0:1]) - torch.min(validatey[:,0:1]))
    validatey[:,1:2] = (validatey[:,1:2] - torch.min(validatey[:,1:2])) / (torch.max(validatey[:,1:2]) - torch.min(validatey[:,1:2]))
    validatey[:,2:3] = (validatey[:,2:3] - torch.min(validatey[:,2:3])) / (torch.max(validatey[:,2:3]) - torch.min(validatey[:,2:3]))

    testx[:,0:1] = (testx[:,0:1] - torch.min(testx[:,0:1])) / (torch.max(testx[:,0:1]) - torch.min(testx[:,0:1]))
    testx[:,1:2] = (testx[:,1:2] - torch.min(testx[:,1:2])) / (torch.max(testx[:,1:2]) - torch.min(testx[:,1:2]))
    testx[:,2:3] = (testx[:,2:3] - torch.min(testx[:,2:3])) / (torch.max(testx[:,2:3]) - torch.min(testx[:,2:3]))

    testy[:,0:1] = (testy[:,0:1] - torch.min(testy[:,0:1])) / (torch.max(testy[:,0:1]) - torch.min(testy[:,0:1]))
    testy[:,1:2] = (testy[:,1:2] - torch.min(testy[:,1:2])) / (torch.max(testy[:,1:2]) - torch.min(testy[:,1:2]))
    testy[:,2:3] = (testy[:,2:3] - torch.min(testy[:,2:3])) / (torch.max(testy[:,2:3]) - torch.min(testy[:,2:3]))
    #testx = (testx - torch.min(testx)) / (torch.max(testx) - torch.min(testx))
    #testy = (testy - torch.min(testy)) / (torch.max(testy) - torch.min(testy))
    
    N, D_in, H, D_out = 64, 9, 100, 3
    model = TwoLayerNet(D_in, H, D_out)
    criterion = torch.nn.MSELoss(reduction='sum')
    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    size = trainx.shape[0]
    batch_size = 1000
    iter_ = int(size / batch_size)
    #print(iter_)
    
    for t in range(500):
        # Forward pass: Compute predicted y by passing x to the model
        #y_pred = model.forward(trainx[0:1000, :].float())
        for i in range(0,iter_):
            range_a = (iter_-1) * batch_size
            range_b = (iter_) * batch_size
            y_pred = model.forward(trainx[range_a:range_b,:].float())
            #y_pred = model.forward(trainx.float())

            # Compute and print loss
            
            loss = criterion(y_pred[range_a:range_b, :], trainy[range_a, range_b].float()) 
            #loss = criterion(y_pred, trainy.float()) 
            print(t, loss.item())

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print('validate:\t,')
    y_pred = model(validatex.float())
    loss = criterion(y_pred, validatey.float())
    print(loss.item())
    print('test:\t,')
    y_pred = model(testx.float())
    loss = criterion(y_pred, testy.float())
    print(loss.item())
