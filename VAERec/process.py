import os
import shutil
import sys

import pickle
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd
import bottleneck as bn
import seaborn as sn
sn.set()


def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    # [movidid,count]
    return count


def filter_triplets(tp, min_uc=5, min_sc=0):
    # Only keep the triplets for items which were clicked on by at least min_sc users.
    if min_sc > 0:
        itemcount = get_count(tp, 'movieId')
        tp = tp[tp['movieId'].isin(itemcount.index[itemcount >= min_sc])]

    # Only keep the triplets for users who clicked on at least min_uc items
    # After doing this, some of the items will have less than min_uc users, but should only be a small proportion
    if min_uc > 0:
        usercount = get_count(tp, 'userId')
        tp = tp[tp['userId'].isin(usercount.index[usercount >= min_uc])]

    # Update both usercount and itemcount after filtering
    usercount, itemcount = get_count(tp, 'userId'), get_count(tp, 'movieId')
    return tp, usercount, itemcount


'''
train, test ,validation 各自内部如何处理，外部如何处理要弄明白
'''

def split_train_test_proportion(data, test_prop=0.2):
    data_grouped_by_user = data.groupby('userId')
    tr_list, te_list = list(), list()

    np.random.seed(98765)

    for i, (_, group) in enumerate(data_grouped_by_user):
        n_items_u = len(group)


        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype='bool')
            idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True

            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])
        else:
            tr_list.append(group)

        if i % 1000 == 0:
            print("%d users sampled" % i)
            sys.stdout.flush()

    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)

    return data_tr, data_te

def numerize(tp):
    uid = map(lambda x: profile2id[x], tp['userId'])
    #print(uid)
    sid = map(lambda x: show2id[x], tp['movieId'])
    #print("-----------------------")
    #print(sid)
    return pd.DataFrame(data={'uid': list(uid), 'sid': list(sid)}, columns=['uid', 'sid'])

DATA_DIR = 'data/ml-20m/'
raw_data = pd.read_csv(os.path.join(DATA_DIR, 'ratings.csv'), header=0)
raw_data = raw_data[raw_data['rating'] > 3.5]

raw_data, user_activity, item_popularity = filter_triplets(raw_data)
# raw_data 格式和原来一样，user_activity [user,count], item_popolarity [item,count]
sparsity = 1. * raw_data.shape[0] / (user_activity.shape[0] * item_popularity.shape[0])
print(sparsity)
#上面求的是数据的稀疏性

# 下面的是把 所有 user_id 打乱顺序
unique_uid = user_activity.index
np.random.seed(98765)
idx_perm = np.random.permutation(unique_uid.size)
unique_uid = unique_uid[idx_perm]

n_users = unique_uid.size
n_heldout_users = 10000

tr_users = unique_uid[:(n_users - n_heldout_users * 2)] #[116677]
vd_users = unique_uid[(n_users - n_heldout_users * 2): (n_users - n_heldout_users)] #[10000]
te_users = unique_uid[(n_users - n_heldout_users):] #[10000]

train_plays = raw_data.loc[raw_data['userId'].isin(tr_users)]
# 格式和raw_data 一样，只不过user_id只在train中出现
unique_sid = pd.unique(train_plays['movieId'])
# 得到独一无二的item_id，去除那些重复的

#构建两个字典，一个是从 item_id 到 id的，另一个是从 user_id到id的
show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))
pro_dir = os.path.join(DATA_DIR, 'pro_sg')

# 将所有独一无二的item_id存起来
'''
with open("processed/user_dict","wb") as f:
    pickle.dump(profile2id,f)
with open("processed/item_id","wb") as f:
    pickle.dump(show2id,f)
'''
vad_plays = raw_data.loc[raw_data['userId'].isin(vd_users)]
vad_plays = vad_plays.loc[vad_plays['movieId'].isin(unique_sid)]
vad_plays_tr, vad_plays_te = split_train_test_proportion(vad_plays)

test_plays = raw_data.loc[raw_data['userId'].isin(te_users)]
test_plays = test_plays.loc[test_plays['movieId'].isin(unique_sid)]
test_plays_tr, test_plays_te = split_train_test_proportion(test_plays)

train_data = numerize(train_plays)
train_data.to_csv("processed/train.csv", index=False)

vad_data_tr = numerize(vad_plays_tr)
vad_data_tr.to_csv("processed/validation_tr.csv", index=False)

vad_data_te = numerize(vad_plays_te)
vad_data_te.to_csv("processed/validation_te.csv", index=False)

test_data_tr = numerize(test_plays_tr)
test_data_tr.to_csv("processed/test_tr.csv", index=False)

test_data_te = numerize(test_plays_te)
test_data_te.to_csv("processed/test_te.csv", index=False)