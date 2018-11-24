from scipy import sparse
import pickle
import pandas as pd
import numpy as np

class trainLoader():
    def __init__(self,dir):
        with open("processed/item_id",'rb') as f:
            item_id=pickle.load(f)
        n_items=len(item_id)
        tp = pd.read_csv(dir)
        n_users = tp['uid'].max() + 1

        rows, cols = tp['uid'], tp['sid']
        data = sparse.csr_matrix((np.ones_like(rows),
                                  (rows, cols)), dtype='float64',
                                 shape=(n_users, n_items))
        self.data=data
        self.n_users=n_users
        self.n_items=n_items

    def get_batch(self,indexs):
        batch=[]
        for index in indexs:
            batch.append(self.data[index].toarray())

        return np.array(batch)

class valLoader():
    def __init__(self,csv_file_tr,csv_file_te):
        with open("processed/item_id",'rb') as f:
            item_id=pickle.load(f)
        n_items=len(item_id)

        tp_tr = pd.read_csv(csv_file_tr)
        tp_te = pd.read_csv(csv_file_te)

        start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
        end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

        rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
        rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

        data_tr = sparse.csr_matrix((np.ones_like(rows_tr),
                                     (rows_tr, cols_tr)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
        data_te = sparse.csr_matrix((np.ones_like(rows_te),
                                     (rows_te, cols_te)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))

        self.data_tr=data_tr
        self.data_te=data_te
        self.n_items=n_items
        self.n_users=end_idx - start_idx + 1
    def get_batch(self,indexs):
        batch=[]
        target=[]
        for index in indexs:
            batch.append(self.data_tr[index].toarray())
            target.append(self.data_te[index].toarray())
        return np.array(batch),np.array(target)
