# Final Project Group 4
- Luke Bogacz
- Vishal Sinha
- Jason Long

## Fannie Mae Loan Performance

## Setup
- Use Python **3.5** or greater.
- Requirements: Use the `requirements.txt` file for necessary dependencies `pip install -r requirements.txt`
- The data is stored in a remote PostGreSQL database, so fetching takes longer than local.
- You can use local data for testing, its located in [reference](./reference), but there is no designated dataloader.

## Running the Models:
The `main.py` located [here](./main.py) is where the models are ran/tested. Under the `if __name__ == '__main__':` block.
At the top there is a `STOP_EARLY` variable, this stops the dataloader iterations early, for the sake of time since the dataset is a bit large and SQL is slow.
This is the only code that is necessary to run the models, it will pull the data from a PostGres database on google cloud. 

## Pre-Processing
The data is pre-processed in two steps:
1. numerical data  partially fitted during loading into the database as `StandardScaler`
2. After all the data is loaded:
   - `OneHotEncoder` is used against the categorical data
   - The target `sdq` is encoded as `0/1`
   - additional info on this: https://medium.com/@contactsunny/label-encoder-vs-one-hot-encoder-in-machine-learning-3fc273365621

- The pre-processing was conducted using these [helpers](./lib/helpers)
- All encoders are pickled and placed in the [pickles](./pickles) dir
- They are then loaded and used to fit the data during `DataLoader` iterations
the production encoder begins with `LIVE.` [here](./pickles/LIVE.pre_processing_encoders.pkl)

## Feature selection
Features were explored as samples of the large dataset using a [jupyter notebook](./feature_selection.ipynb), a script version is located [here](./feature_selection.py)

## Loading Data
The data loading logic is located in [here](./lib/data/loader.py). This file is what fetches the data from the remote SQL table, applies normalization, and returns it as a tensor.

Below is an example for loading data:
```python
from torch.utils.data import DataLoader
from lib.db import connect
from lib.data.loader import LoanPerformanceDataset 

dataset = LoanPerformanceDataset(
    chunk=10,  # size of the query (use a large number here)
    conn=connect(local=False).connect(), # connect to remote database instance ( google cloud )
    ignore_headers=['co_borrower_credit_score_at_origination'],
    target_column='sdq'
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