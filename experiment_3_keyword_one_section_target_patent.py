import pickle
from time import time
import numpy as np
import pandas as pd
from scipy.sparse import save_npz, load_npz
import os
import whoosh_utils
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from collections import defaultdict,Counter
import re

import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for headless environments
import matplotlib.pyplot as plt

whoosh_index = whoosh_utils.load_index('validation/validation_index')
searcher = whoosh_utils.get_searcher(whoosh_index)
qp = whoosh_utils.get_query_parser()



def load_tfidf(field, path):
    tfidf_scores = {}
    if field == 'description':
        with open(path,'rb') as file:
            tfidf_scores = pickle.load(file)
            file.close()
            return tfidf_scores
    elif field == "cpc":
        with open(path,'rb') as file:
            cpc_codes = pickle.load(file)
            file.close()
        test_file_path = "validation/neighbors_small.csv" #Change it to submission.csv in kagggle submission
        test_set = pd.read_csv(test_file_path).values
        test_set = test_set.tolist()

        test_cpc = {}
        for row in test_set:
            cnt = defaultdict(lambda: 0)
            for patent in row:
                for cpc in cpc_codes[patent]:
                    cnt[cpc]+=1
            #cnt = sorted(cnt.items(),key=lambda x: x[1])

            for patent in row:
                test_cpc[patent] = sorted([(cpc,cnt[cpc]) for cpc in cpc_codes[patent]], key=lambda x:x[1], reverse=True)
        return test_cpc

    else:
        cache = path
        tfidf_matrix = load_npz(os.path.join(cache, field+'_tfidf_matrix.npz'))
        feature_names = np.load(os.path.join(cache, field+'_feature_names.npz'), allow_pickle=True)['feature_names']
        #print("Feature Names:",len(feature_names))
        publication_numbers = np.load(os.path.join(cache, field+'_publication_numbers.npy'), allow_pickle=True)
        # Create a mapping from publication number to index for fast look-up
        publication_index_map = {pub: idx for idx, pub in enumerate(publication_numbers)}
        return (tfidf_matrix, feature_names, publication_index_map)

start = time()
#Loading TFIDF matrix
tfidf_data = {
    "title": load_tfidf("title", "cache/tfidf"),
    "abstract": load_tfidf("abstract", "cache/tfidf"),
    "claims": load_tfidf("claims", "cache/tfidf"),
    "description": load_tfidf("description", "cache/tfidf/description/test_description_200000_tfidf_all_keywords_l2_normalized.pickle"),
    "cpc": load_tfidf("cpc", "cache/cpc_codes.pickle")
    }


end = time()
execution_time = (end - start)  # In second
print(f"TFIDF Loading Time: {execution_time:.2f} second")

def count_query_tokens(query: str):
    # Count the number of tokens in the query.
    # Treat entries like "cpc:AO1B33/00" as a single token.
    return len([i for i in re.split('[\s+()]', query) if i])

def evaluate_row(args):
    labels = args[0][1:]
    query = args[1][1]
    #qp = args[2]
    #searcher = args[3]
    preds = whoosh_utils.execute_query(query, qp, searcher)
    precisions = list()
    n_label = len(labels)
    n_found = 0
    for e, i in enumerate(preds):
        if i in labels:
            n_found += 1
            precisions.append(n_found/(e+1)) # this is how it probably should be

    #return sum(precisions)/50,args[0][0],len(preds),n_found
    return sum(precisions)/min(len(labels), 50),args[0][0],len(preds),n_found,count_query_tokens(query), preds

def extract_keywords(patent, field, top_n=100):
    try:
        if field== "description":
            top_n_keys = tfidf_data[field][patent][:top_n]
            '''for item in tfidf_data[field][patent][top_n:]:
                if description_df[item[0]] == 1:
                    top_n_keys.append(item)'''
            return top_n_keys
            #return tfidf_data[field][patent]
        elif field=="cpc":
            top_n_keys = tfidf_data[field][patent][:top_n]
            return top_n_keys
        else:
            doc_index = tfidf_data[field][2][patent]
            vector = tfidf_data[field][0][doc_index].toarray().flatten()
            if top_n < len(vector):
                top_indices = np.argpartition(vector, -top_n)[-top_n:]
            else:
                top_indices = np.argsort(vector)
            return sorted([(tfidf_data[field][1][idx], vector[idx]) for idx in top_indices if vector[idx]!=0], key=lambda item: item[1], reverse=True)
            #return [(tfidf_data[field][1][idx], vector[idx]) for idx in range(len(vector)) if vector[idx]!=0]
    except:
        return []


def expression_reducer(expr):
    one_token = [item[0] for item in expr if len(item)==1]
    two_token = [item for item in expr if len(item)==2]

    query = []
    while len(two_token)!=0:
        token_count = defaultdict(set)
        for pair in two_token:
            token_count[pair[0]].add(pair[1])
            token_count[pair[1]].add(pair[0])
        term = sorted(token_count.items(),key =lambda x: len(x[1]),reverse=True)[0]

        query.append("("+term[0]+" ("+" OR ".join(term[1])+"))")
        for item in term[1]:
            if (term[0],item) in two_token:
                two_token.remove((term[0],item))
            elif (item,term[0]) in two_token:
                two_token.remove((item,term[0]))
    q = []
    if len(query)!=0:
        q.append(" OR ".join(query))
    if len(one_token)!=0:
        q.append(" OR ".join(one_token))
    return " OR ".join(q)


#Only target patent
'''def build_query_for_row(test_row, field1, field2):
    #********Change logic here***************
    prefix = {
        "title":"ti:", 
        "abstract":"ab:", 
        "claims":"clm:", 
        "description":"detd:", 
        "cpc":"cpc:"
        }

    combined_keywords = {}
    combined_keywords_pairs = {}
    top_3_keys = {}
    pairs = []
    for neighbor in test_row[1:]:
        #field1 = "description"
        #field2 = "cpc_codes"
        #field = "title"
        #extracted_keys = extract_keywords(neighbor, field, top_n=100)
        #top_3=tuple(sorted([prefix[field]+k for k,v in extracted_keys[:2]]))
        top_field1 = extract_keywords(neighbor, field1, top_n=100)
        #top_field2 = extract_keywords(neighbor, field2, top_n=100)
        #print(top_field1)
        #print(top_field1)
        #for index in range(0, max(len(top_field1),len(top_field2))-1):
        for index in range(0, len(top_field1)):
            #if index == len(top_field1)-1 or len(top_field1)==0:
            #    break
            if index<len(top_field1)-1:
                pairs.append((prefix[field1]+top_field1[index][0],prefix[field1]+top_field1[index+1][0]))
                #pairs.append((prefix[field1]+top_field1[index][0],))
            #if index<len(top_field2)-1:
                #pairs.append((prefix[field2]+top_field2[index][0],prefix[field2]+top_field2[index+1][0]))
                #pairs.append((prefix[field1]+top_field1[index][0],))

        #top_field2 = extract_keywords(neighbor, field2, top_n=100)
        top_field2 = []
        top_3 = ""
        if len(top_field1)==0:
            top_3=tuple(sorted([prefix[field2]+k for k,v in top_field2[:2]]))
        elif len(top_field2)==0:
            top_3=tuple(sorted([prefix[field1]+k for k,v in top_field1[:2]]))
        elif len(top_field1)==0 and len(top_field2)==0:
            continue
        elif top_field1[0][0] == top_field2[0][0]:
            if top_field1[0][1]>top_field2[0][1]:
                top_3 = tuple([prefix[field1]+top_field1[0][0]])
            else:
                top_3 = tuple([prefix[field2]+top_field2[0][0]])
        else:
            top_3 = tuple([prefix[field1]+top_field1[0][0], prefix[field2]+top_field2[0][0]])

        if top_3 in top_3_keys:
            top_3_keys[top_3]+=1
        else:
            top_3_keys[top_3]=1'''
    
    #sorted_top_3_keys = [k for k,v in sorted(top_3_keys.items(), key=lambda item: item[1], reverse=True)]

    #print(pairs)  
    #return test_row[0], pairs


#Only neighbours one section
'''def build_query_for_row(test_row, field1, field2):
    #********Change logic here***************
    prefix = {
        "title":"ti:", 
        "abstract":"ab:", 
        "claims":"clm:", 
        "description":"detd:", 
        "cpc":"cpc:"
        }

    combined_keywords = {}
    combined_keywords_pairs = {}
    top_3_keys = {}
    pairs = []
    for neighbor in test_row[1:]:
        top_field1 = extract_keywords(neighbor, field1, top_n=100)
        pairs.append(tuple(sorted([prefix[field1]+k for k,v in top_field1[:2]])))
    count = Counter(pairs)
    sorted_tuples = [item[0] for item in sorted(count.items(), key=lambda x: x[1], reverse=True)]
    return test_row[0], sorted_tuples'''

#Top 1 from two section (two keyword combined)
def build_query_for_row(test_row, field1, field2):
    #********Change logic here***************
    prefix = {
        "title":"ti:", 
        "abstract":"ab:", 
        "claims":"clm:", 
        "description":"detd:", 
        "cpc":"cpc:"
        }

    combined_keywords = {}
    combined_keywords_pairs = {}
    top_2_keys = {}
    for neighbor in test_row[1:]:
        #field1 = "description"
        #field2 = "cpc_codes"
        #field = "title"
        #extracted_keys = extract_keywords(neighbor, field, top_n=100)
        #top_2=tuple(sorted([prefix[field]+k for k,v in extracted_keys[:2]]))
        top_field1 = extract_keywords(neighbor, field1, top_n=100)
        #top_field2 = extract_keywords(neighbor, field2, top_n=100) #For one section comment this and uncomment the next line
        top_field2 = [] #For one section uncomment this and comment the previous line
        top_2 = ""

        if len(top_field1)==0:
            top_2=tuple(sorted([prefix[field2]+k for k,v in top_field2[:2]]))
        elif len(top_field2)==0:
            top_2=tuple(sorted([prefix[field1]+k for k,v in top_field1[:2]]))
        elif len(top_field1)==0 and len(top_field2)==0:
            continue
        elif top_field1[0][0] == top_field2[0][0]:
            if top_field1[0][1]>top_field2[0][1]:
                top_2 = tuple([prefix[field1]+top_field1[0][0]])
            else:
                top_2 = tuple([prefix[field2]+top_field2[0][0]])
        else:
            top_2 = tuple([prefix[field1]+top_field1[0][0], prefix[field2]+top_field2[0][0]])

        if top_2 in top_2_keys:
            top_2_keys[top_2]+=1
        else:
            top_2_keys[top_2]=1
    
    sorted_top_2_keys = [k for k,v in sorted(top_2_keys.items(), key=lambda item: item[1], reverse=True)]
    '''query = expression_reducer(sorted_top_2_keys[:17])
    for i in reversed(range(len(sorted_top_2_keys))):
        temp_query = expression_reducer(sorted_top_2_keys[:i])
        if count_query_tokens(temp_query) > 50:
            continue
        else:
            query = temp_query
            break
            
    return test_row[0], query'''
            
    return test_row[0], sorted_top_2_keys


'''def build_query_for_row(test_row):
    #********Change logic here***************
    prefix = {
        "title":"ti:", 
        "abstract":"ab:", 
        "claims":"clm:", 
        "description":"detd:", 
        "cpc":"cpc:"
        }

    top_3_keys = {}
    key_neighbors_coverage = defaultdict(set)

    for neighbor in test_row[1:]:
        field1 = "title"
        field2 = "cpc"
        field3 = "description"
        #field = "title"
        #extracted_keys = extract_keywords(neighbor, field, top_n=100)
        #top_3=tuple(sorted([prefix[field]+k for k,v in extracted_keys[:2]]))
        top_field1 = extract_keywords(neighbor, field1, top_n=100)
        top_field2 = extract_keywords(neighbor, field2, top_n=100)
        top_3 = ""
        if len(top_field1)==0:
            top_3=tuple(sorted([prefix[field2]+k for k,v in top_field2[:2]]))
        elif len(top_field2)==0:
            top_3=tuple(sorted([prefix[field1]+k for k,v in top_field1[:2]]))
        elif len(top_field1)==0 and len(top_field2)==0:
            continue
        elif top_field1[0][0] == top_field2[0][0]:
            if top_field1[0][1]>top_field2[0][1]:
                top_3 = tuple([prefix[field1]+top_field1[0][0]])
            else:
                top_3 = tuple([prefix[field2]+top_field2[0][0]])
        else:
            top_3 = tuple([prefix[field1]+top_field1[0][0], prefix[field2]+top_field2[0][0]])

        if top_3 in top_3_keys:
            top_3_keys[top_3]+=1
        else:
            top_3_keys[top_3]=1
        key_neighbors_coverage[top_3].add(neighbor)
    
    sorted_top_3_keys = [k for k,v in sorted(top_3_keys.items(), key=lambda item: item[1], reverse=True)]
    query = expression_reducer(sorted_top_3_keys[:17])
    slectedUntill = 17
    for i in reversed(range(len(sorted_top_3_keys))):
        temp_query = expression_reducer(sorted_top_3_keys[:i+1])

        if count_query_tokens(temp_query) > 46:
            continue
        else:
            query = temp_query
            slectedUntill = i+1
            break
    
    #Select additional item for the element which are not covered
    covered_nighbor = set()
    for item in sorted_top_3_keys[:slectedUntill]:
        covered_nighbor.update(key_neighbors_coverage[item])
    top_2_field3 = defaultdict(lambda: 0)
    for n in test_row[1:]:
        if n not in covered_nighbor:
            for k,v in extract_keywords(neighbor, field3, top_n=100):
                top_2_field3[k]+=v
    top_2_field3 = sorted(top_2_field3.items(),key=lambda k:k[1], reverse=True)
    if len(top_2_field3[:2])>0:
        query +=" OR (" +" ".join([item[0] for item in top_2_field3[:3]])+")"
        
    return test_row[0], query'''




if __name__ == "__main__":
    testSize = 2500
    test_file_path = "validation/neighbors_small.csv" #Change it to submission.csv in kagggle submission
    #test_set = pd.read_csv(test_file_path, nrows=testSize).values #Remove nrows in kaggle submission
    test_set = pd.read_csv(test_file_path).values #Remove nrows in kaggle submission
    #test_set = pd.read_csv(test_file_path).tail(1500).values #Remove nrows in kaggle submission
    all_patents_test_set = set(test_set.flatten())
    test_set = test_set.tolist()
    max_workers = max(1, os.cpu_count() // 4)
    
    '''combinations = [
        ("title", "abstract"),
        ("title", "claims"),
        ("title", "description"),
        ("title", "cpc"),
        ("abstract", "claims"),
        ("abstract", "description"),
        ("abstract", "cpc"),
        ("claims", "description"),
        ("claims", "cpc"),
        ("description", "cpc")
    ]'''

    combinations = [
        ("title", "title"),
        ("abstract", "abstract"),
        ("claims", "claims"),
        ("description", "description"),
        ("cpc", "cpc")
    ]

    all_combination_results = {}
    for field1,field2 in combinations:
        print(f"{field1} - {field2}")
        #Build query for each rows
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            submission_queries = list(tqdm(executor.map(build_query_for_row, test_set, [field1]*len(test_set),[field2]*len(test_set)), total=len(test_set), desc="Building Queries"))

        pd.DataFrame(submission_queries, columns=['publication_number', 'query']).to_csv("submission.csv", index=False)

        ap_result = []
        for i in [5, 10,20,30,40,50]:
            queries = [(patent, expression_reducer(sorted_top_2_keys[:i])) for patent, sorted_top_2_keys in submission_queries]

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = list(tqdm(executor.map(evaluate_row, list(zip(test_set, queries))), total=len(test_set), desc="Executing Queries"))
                ap = [item[0] for item in results]
                pred = [item[5] for item in results]
                query_length = [item[4] for item in results]
                retrival_cnt = [(item[1],item[0],item[2],item[3],item[4]) for item in results]
            
            ap_result.append((np.mean(query_length),np.mean(ap)))
            pd.DataFrame({
                "publication_number": [item[0] for item in queries],
                "query": [item[1] for item in queries],
                "query_length": query_length,
                "ap": ap,
                "pred_retrival_len": [len(x) for x in pred],
                "gold": [", ".join(x[1:]) for x in test_set],
                "predict": [", ".join(x) for x in pred]}).to_excel(f"result_one_section_temp/{field1}-{field2}_k{np.mean(query_length)}.xlsx", index=False)
        #print("Avg_length", "AP")
        #print(ap_result)
        all_combination_results[field1+" - "+field2]=ap_result

        #*************Save as line plot*************************
        # Unpack the list of tuples into two separate lists for x and y values


        #pd.DataFrame(retrival_cnt, columns=["publication_number", "AP","retrived_item", "relevent__item_retrived","query_length"]).to_csv("n_match_test_set.csv", index=False)
        
        #print("Query length:",whoosh_utils.count_query_tokens(submission_queries[25][1]))
        #print(qp.parse(submission_queries[25][1]))
    print(all_combination_results)