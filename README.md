## Fannie Mae Loan Performance

## Setup
- Use the `requirements.txt` file for necessary dependencies `pip install -r requirements.txt`
- The data is stored in a remote PostGreSQL database, so fetching takes longer than average.
- You can use local data for testing, its located in [reference](./reference)

## Pre-Processing
The data is pre-processes in two steps:
1. numerical data  partially fitted during loading into the database as `StandardScaler`
2. After all the data is loaded:
   - `OneHotEncoder` is used against the categorical data
   - The target 'current_loan_delinquency_status' is encoded using `LabelEncoder`
   - additional info on this: https://medium.com/@contactsunny/label-encoder-vs-one-hot-encoder-in-machine-learning-3fc273365621
   
All encoders are pickled and placed in the [pickles](./pickles) dir
They are then loaded and used to fit the data during `DataLoader` iterations

## Loading Data
The data loading logic is located in [here](./lib/data/loader.py)

Below is an example for loading data:
```python
from torch.utils.data import DataLoader
from lib.db import connect
from lib.data.loader import LoanPerformanceDataset 

dataset = LoanPerformanceDataset(
    chunk=10,  # size of the query (use a large number here)
    conn=connect(local=False).connect(), # connect to remote database instance ( google cloud )
    # conn=connect(local=True).connect(), # connect to local database (docker)
    ignore_headers=['loan_id'],
    target_column='current_loan_delinquency_status'
)
loader = DataLoader(
    dataset,
    batch_size=1,  # size of batches from the query, 1 === size of query, 2 = 1/2 query size
    num_workers=1,
    shuffle=False
)

# your epoch iteration here..
for batch_idx, (features, targets) in enumerate(loader):
    print('batch: {} size: {}'.format(batch_idx, targets.size()))
    print(features.size())
   # send data to model here
    
```