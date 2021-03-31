import pandas as pd
import numpy as np

def tags_to_one_hot(tags):
    # tag to index
    tags_index_dict = dict()
    for i in range(0, len(tags)):
        tags_index_dict[tags["tagID"][i]] = i
    return tags_index_dict

def jobs_company_to_one_hot(jobs):
    # size to one hot
    size_dict = {"1-10": 1, "11-50": 2, "51-100": 3, "101-200": 4, "201-500": 5, "501-1000": 6, "1000 이상": 7}
    a = 1
    for k, v in size_dict.items():
        m = np.zeros(shape=(8))
        m[a] = 1.
        size_dict[k] = m
        a += 1

    # jobs to one hot
    company_one_hot = dict()
    for i in range(0, len(jobs)):
        id = jobs["jobID"][i]
        size = jobs["companySize"][i]
        if size not in size_dict:
            size = np.zeros(shape=(8))
            size[0] = 1.
        else:
            size = size_dict[jobs["companySize"][i]]
        company_one_hot[id] = size
    return company_one_hot

def id_to_one_hot(tags_index_dict, users, index):
    # ids to tag dic
    users_tags_dict= dict()
    for i in range(0, len(users)):
        first = users[index][i]
        tagID = users["tagID"][i]

        if first not in users_tags_dict:
            users_tags_dict[first] = [tagID]
        else:
            k = users_tags_dict[first]
            k.append(tagID)
            users_tags_dict[first] = k

    # ids to tag one hot
    id_one_hot = dict()
    for key, values in users_tags_dict.items():
        # one_hot = [0 for i in range(0, len(tags_index_dict))]
        one_hot = np.zeros(shape=(len(tags_index_dict)))
        for v in values:
            one_hot[tags_index_dict[v]] = 1.
        id_one_hot[key] = one_hot
    return id_one_hot

data_train = pd.read_csv("./datas/train.csv")
data_user_tag = pd.read_csv("./datas/user_tags.csv")
data_tags = pd.read_csv("./datas/tags.csv")
data_jobs = pd.read_csv("./datas/job_tags.csv")
data_companies = pd.read_csv("./datas/job_companies.csv")

tags_one_hot = tags_to_one_hot(data_tags)
users_one_hot = id_to_one_hot(tags_one_hot, data_user_tag, "userID")
jobs_one_hot = id_to_one_hot(tags_one_hot, data_jobs, "jobID")
# company_one_hot = jobs_company_to_one_hot(data_companies)

X = np.empty(shape=(0,1774))
Y = np.asarray(data_train["applied"].tolist())
for i in range(0, len(data_train)):
    all = np.concatenate((users_one_hot[data_train["userID"][i]], jobs_one_hot[data_train["jobID"][i]]), axis=0)
    X = np.append(X, np.array([all]), axis=0)

np.save("./datas/X2_6000", X)
# np.save("./datas/Y1_6000",Y)