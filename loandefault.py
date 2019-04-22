import torch
import torchvision
from torch.autograd import Variable
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import pickle

# download data from MNIST and create mini-batch data loader
torch.manual_seed(1122)
#----------------------------------------------------------------------------
Batch_size = 1     # Batch size
R = 1000            # Input size
S = 10             # Number of neurons
a_size = 1              # Network output size
#----------------------------------------------------------------------------

feature1 =  'ORIGINAL_CLTV','FICO_ORIG'
feature2 = 'ORIGINAL_LOAN_TERM','ORIGINAL_CLTV','FICO_ORIG','ORIGINAL_INTEREST_RATE','ORIGINAL_UPB'
feature3 =  'ORIGINAL_CLTV','FICO_ORIG','ORIGINAL_INTEREST_RATE'

feature4 = 'ORIGINAL_LOAN_TERM','ORIGINAL_CLTV','CBD_CURRENT_FICO','ORIGINAL_INTEREST_RATE','ORIGINAL_UPB','ORIGINAL_DTI'
feature5 = 'ORIGINAL_LOAN_TERM','ORIGINAL_CLTV','CBD_CURRENT_FICO','ORIGINAL_INTEREST_RATE','ORIGINAL_UPB','ORIGINAL_DTI','CURRENT_RATE_INCENTIVE','NUMBER_OF_BORROWERS'
feature6 = 'ORIGINAL_CLTV','CBD_CURRENT_FICO','ORIGINAL_DTI','NUMBER_OF_BORROWERS'
max_loop =300
split_size =0.3
random_seed_exp=45

feature_list=[]
feature_list.append(feature1)
feature_list.append(feature2)
feature_list.append(feature3)
feature_list.append(feature4)
feature_list.append(feature5)
feature_list.append(feature6)

## to be replaced with data loader
print("Reading Loan profile ")
observationData = pd.read_table("Acquisition_2017Q1.txt", header=0,sep='|',keep_default_na=False)
observationData = observationData.reindex(columns=['LOAN','ORIGINATION_CHANNEL','SELLER_NAME',
                                          'ORIGINAL_INTEREST_RATE','ORIGINAL_UPB',
                                          'ORIGINAL_LOAN_TERM',
                                          'ORIGINATION_DATE','FIRST_PAYMENT_DATE','ORIGINAL_LTV',
                                          'ORIGINAL_CLTV','NUMBER_OF_BORROWERS','ORIGINAL_DTI',
                                          'FICO_ORIG','FIRST_TIME_BUYER_INDICATOR','LOAN_PURPOSE,PROPERTY_TYPE',
                                          'NUMBER_OF_UNITS','OCCUPANCY_TYPE','PROPERTY_STATE','ZIP_CODE_SHORT',
                                          'MI','PRODUCT_TYPE','CO-BORROWER_FICO_ORIGINATION','MORTGAGE_INSURANCE_TYPE',
                                          'RELOCATION_MORTGAGE_INDICATOR'])

#print(observationData['LOAN'])

print ("Reading the performance data")
colnames=['LOAN','Month','SERVICER_NAME','CURRENT_INTEREST_RATE','CURRENT_ACTUAL_UPB','LOAN_AGE','REMAINING_MONTHS','ADJUSTED_MONTHS_TO_MATURITY','MATURITY_DATE','MSA','SDQ','MODIFICATION_FLAG','ZERO_BALANCE_CODE','ZERO_BALANCE_EFFECTIVE_DATE','LAST_PAID_INSTALLMENT_DATE','FORECLOSURE_DATE','DISPOSITION_DATE','FORECLOSURE_COSTS',
                                         'PROPERTY_PRESERVATION_COSTS','ASSET_RECOVERY_COSTS','MISCELLANEOUS_HOLDING_EXPENSES_CREDITS',
                                         'ASSOCIATED_TAXES_FOR_HOLDING_PROPERTY','NET_SALE_PROCEEDS','CREDIT_ENHANCEMENT_PROCEEDS',
                                         'REPURCHASE_MAKE_WHOLE_PROCEEDS','OTHER_FORECLOSURE_PROCEEDS','NON_INTEREST_BEARING_UPB','PRINCIPALFORGIVENESS','REPURCHASE','FORECLOSURE_AMOUNT','SERVICING_INDICATOR']
targetData = pd.read_table("Performance_2017Q1.txt",names=colnames, header=0,sep='|',keep_default_na=False)

#print(targetData)
#print(targetData['SDQ'])
#print(targetData.columns)
#for key,value in targetData.iteritems():
 #    print(key,value)


print("Grouping the DF")
targetData2=targetData[['LOAN','SDQ']].groupby('LOAN',as_index=False).sum()


#targetData3 = targetData2['LOAN','SDQ']
#print(targetData2.columns)
targetData2 = targetData2.reindex(columns=['LOAN','SDQ'])
#print(targetData2)

print("Merging the DF")
obs3 = observationData.merge(targetData2, left_on='LOAN', right_on='LOAN')
obs_shuffle = obs3.iloc[np.random.permutation(len(obs3))]
obs2 = obs_shuffle[:1000]
t1 = pd.to_numeric(obs2['SDQ'], errors='coerce')
t2 = t1.replace(np.nan, 0, regex=True)
t2 = (t2>0).astype(int)


targetData2.to_csv("loan_perf.csv")
obs2.to_csv("final.csv")

######

outputModel = 'loandefaulttmp'
finalModel = 'loandefaultfinal'
modelPATH= '/home/ubuntu/Machine-Learning/ml2-project/'
feature_final =0
accuracy_final = 0


def approx_func(x=None):
    ret = torch.sin(x)
    return (ret)


class MLP_sample(nn.Module):

    def __init__(self, h_sizes, out_size):

        super(MLP, self).__init__()

        # Hidden layers
        self.hidden = []
        for k in range(len(h_sizes)-1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k+1]))

        # Output layer
        self.out = nn.Linear(h_sizes[-1], out_size)

    def forward(self, x):

        # Feedforward
        for layer in self.hidden:
            x = F.relu(layer(x))
        output= F.softmax(self.out(x), dim=1)

        return output

# define and initialize a multilayer-perceptron, a criterion, and an optimizer
class MLP(nn.Module):
    def __init__(self,input_size):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(input_size , S)
        self.t1 = nn.ReLU()
        self.l2 = nn.Linear(S,a_size)
        self.t2 = nn.Sigmoid()

    def forward(self, x,input_size):
       #x = x.view(-1, input_size)
        x = self.t1(self.l1(x))
        x = self.t2(self.l2(x))
        return x

class MLP1(nn.Module):
    def __init__(self,input_size):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(input_size , S)
        self.t1 = nn.ReLU()
        self.l2 = nn.Linear(S,S)
        self.t2 = nn.ReLU()
        self.l3 = nn.Linear(S,a_size)
        self.t3 = nn.Softmax()

    def forward(self, x,input_size):
       #x = x.view(-1, input_size)
        x = self.t1(self.l1(x))
        x = self.t2(self.l2(x))
        x = self.t3(self.l3(x))

        return x


# --------------------------------------------------------------------------------------------

performance_index = torch.nn.MSELoss(size_average=False)

# --------------------------------------------------------------------------------------------

# define a training epoch function
def trainEpoch(epoch,p,q,input_size):
    #print("Training Epoch %i" % (epoch + 1))
    #print("Input Size ",input_size)
    mlp = MLP(input_size)
    optimizer = optim.SGD(mlp.parameters(), lr=0.1, momentum=0.9)
    mlp.train()
    running_loss = 0
    optimizer.zero_grad()

    outputs = mlp(p,input_size)
    #print(torch.round(outputs))
    loss = performance_index(torch.round(outputs), q)
    #loss = criterion(outputs, t)
    loss.backward()
    optimizer.step()
    running_loss = loss.item()

    return running_loss , mlp

# --------------------------------------------------------------------------------------------

def convert_pd_tensor(p_pd):
    pnumpy = p_pd.fillna(0)
    pnumpy2 = pd.to_numeric(pnumpy, errors='coerce')
    pnumpy3 = pnumpy2.replace(np.nan, 0, regex=True)
    pnumpy4 = pnumpy3.replace('', 0, regex=True)
    pnumpy5 = pnumpy4.values
    pnumpy6 = pnumpy5.astype(np.float32)
    return (torch.from_numpy(pnumpy6))

for k in range(0,len(feature_list)):
    row_size = obs2.shape[0]
    #print("Row size ",row_size)
    p = Variable(torch.randn(row_size, len(feature_list[k])))
    t = Variable(torch.randn(row_size, a_size), requires_grad=False)

    #p1  =  torch.from_numpy(pnumpy6)
    p = convert_pd_tensor(obs2.loc[:, feature_list[k]])
    t = convert_pd_tensor(t2)
    #t = torch.tensor(obs2['SDQ'].values)
    #q = obs2[['SDQ']]
    print(feature_list[k])
    #print(p)
    #print(t)

    #t = np.squeeze(targetData2.loc[:, 'SDQ'].values)
    #X_train, X_test, Y_train, Y_test = train_test_split(p, t, test_size=split_size,random_state=random_seed_exp,shuffle=False)

    accuracy =0
    avg_accuracy_prev =0
    avg_accuracy_curr =0
    tmp_accuracy_prev =-1000000
    tmp_accuracy_curr =-1000000
    change=0


    for epoch in range(max_loop):

            print ('Iter ',epoch)
            #print('Training')
            tmp_accuracy_curr,currmodel = trainEpoch(epoch,p,t,len(feature_list[k]))
            print('[%d] loss: %.3f' %
                  (epoch + 1, tmp_accuracy_curr))
            accuracy += tmp_accuracy_curr
            if(abs(tmp_accuracy_curr)< abs(tmp_accuracy_prev)):
                    tmp_accuracy_prev=tmp_accuracy_curr
                    change=epoch
                    saveModle = outputModel+"_"+str(k)+str(epoch)
                    torch.save(currmodel,modelPATH+saveModle)
                    print('Save Model: ' , saveModle)
    avg_accuracy_curr = tmp_accuracy_prev

    print('Average Accuracy %.2f' % avg_accuracy_curr)

    if (avg_accuracy_prev==0):
          avg_accuracy_prev = avg_accuracy_curr
    if(avg_accuracy_curr < avg_accuracy_prev):
          avg_accuracy_prev = avg_accuracy_curr


    if (accuracy_final == 0):
        accuracy_final = avg_accuracy_prev
        finalModel = saveModle
    if (avg_accuracy_prev < accuracy_final):
        accuracy_final = avg_accuracy_prev
        feature_final = k
        finalModel = saveModle

print ('Final Accuracy %.2f' % accuracy_final )
print ('Features Selected ' ,feature_list[feature_final] )


# run the training epoch 30 times and test the result
#for epoch in range(100):
    #trainEpoch( epoch)


