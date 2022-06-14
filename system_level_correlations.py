
import json
import argparse
import math
from collections import defaultdict
import numpy as np


parser = argparse.ArgumentParser(description='Process argument')
parser.add_argument('--correlator_str', default='kendall', help='which correlation metric to use')
parser.add_argument('--only_extractive', action='store_true', \
    help='only include extractive methods in calculations')
parser.add_argument('--annotators_str', default='expert_annotations', \
    help='string specificying whether to use expert annotations or turker annotations')
parser.add_argument('--only_abstractive', action='store_true', \
    help='only include abstractive methods in calculations')
parser.add_argument('--subset', default=11, \
    help='how many references used to calculate metric scores for correlation calculations')
parser.add_argument('--input_file', \
    default="model_annotations.aligned.scored.jsonl", \
    help="jsonl file with annotations and metric scores")
args = parser.parse_args()
assert not (args.only_extractive and args.only_abstractive)
if args.correlator_str == "pearson":
    from scipy.stats import pearsonr as correlator
else:
    from scipy.stats import kendalltau as correlator

tmp_summ_ids = {'M15', 'M20', 'M0', 'M10', 'M23_C4', 'M23_dynamicmix', 'M2', \
    'M5', 'M17', 'M11', 'M9', 'M22', 'M12', 'M1', 'M8', 'M13', 'M14'}
extractive_ids = ["M0", "M1", "M2", "M5"]
abstractive_ids = list(tmp_summ_ids - set(extractive_ids))

summ_ids = set()

sorted_keys = ['rouge_1_f_score', 'rouge_2_f_score', 'rouge_3_f_score', \
    'rouge_4_f_score', 'rouge_l_f_score', 'rouge_su*_f_score', \
    'rouge_w_1.2_f_score', 'rouge_we_1_f', 'rouge_we_2_f', 'rouge_we_3_f', \
    's3_pyr', 's3_resp', 'bert_score_precision', 'bert_score_recall', \
    'bert_score_f1', 'mover_score', 'sentence_movers_glove_sms', 'summaqa_avg_fscore', \
    'blanc', 'supert', 'bleu', 'chrf', 'cider', \
    'meteor', 'summary_length', 'percentage_novel_1-gram', \
    'percentage_novel_2-gram', 'percentage_novel_3-gram', \
    'percentage_repeated_1-gram_in_summ', 'percentage_repeated_2-gram_in_summ', \
    'percentage_repeated_3-gram_in_summ', 'coverage', 'compression', 'density']
table_names = ['ROUGE-1 ', 'ROUGE-2 ', 'ROUGE-3  ', 'ROUGE-4 ', 'ROUGE-L  ', \
    'ROUGE-su* ', 'ROUGE-w  ', 'ROUGE-we-1 ', \
    'ROUGE-we-2 ', 'ROUGE-we-3  ', '$S^3$-pyr ', '$S^3$-resp  ', \
    'BertScore-p ', 'BertScore-r ', 'BertScore-f  ', 'MoverScore ', \
    'SMS  ', 'SummaQA\\^ ', 'BLANC', 'SuPERT', 'BLEU  ', 'CHRF  ', 'CIDEr  ', \
    'METEOR  ', 'Length\\^  ', 'Novel unigram\\^ ', \
    'Novel bi-gram\\^ ', 'Novel tri-gram\\^  ', 'Repeated unigram\\^ ', \
    'Repeated bi-gram\\^ ', 'Repeated tri-gram\\^  ', \
    'Stats-coverage\\^ ', 'Stats-compression\\^ ', 'Stats-density\\^ ']

article2humanscores = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
article2systemscores = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

metrics = set()
articles = set()
with open(args.input_file) as inputf:
    for line_count, line in enumerate(inputf):
        # if line_count >= 1600:
        #     break
        data = json.loads(line)
        curid = data['id']
        summ_id = data['model_id']
        if args.only_extractive and summ_id not in extractive_ids:
            continue
        if args.only_abstractive and summ_id not in abstractive_ids:
            continue

        summ_ids.add(summ_id)
        articles.add(curid)

        annotations = data[args.annotators_str]
        coh = [x["coherence"] for x in annotations]
        con = [x["consistency"] for x in annotations]
        flu = [x["fluency"] for x in annotations]
        rel = [x["relevance"] for x in annotations]
        #annotations_mean = np.mean(annotations, axis=0).tolist()
        article2humanscores[curid][summ_id]["coherence"] = np.mean(coh)
        article2humanscores[curid][summ_id]["consistency"] = np.mean(con)
        article2humanscores[curid][summ_id]["fluency"] = np.mean(flu)
        article2humanscores[curid][summ_id]["relevance"] = np.mean(rel)

        scores = data[f'metric_scores_{args.subset}']
        for key1, val1 in scores.items():
            if key1 == "id":
                continue
            # supert returned a list of length 1
            if key1 == "supert":
                article2systemscores[curid][summ_id][key1] = val1[0]
                metrics.add(key1)
            elif key1 == "rouge":
                for key2, val2 in scores["rouge"].items():
                    article2systemscores[curid][summ_id][key2] = val2
                    metrics.add(key2)
            else:
                article2systemscores[curid][summ_id][key1] = val1
                metrics.add(key1)

summ_ids = list(summ_ids)
summ_ids = sorted(summ_ids, key=lambda x: int("".join([i for i in x if i.isdigit()])))
metric2table = {}
metrics = ['rouge_1_f_score','rouge_2_f_score','rouge_3_f_score','rouge_4_f_score','rouge_l_f_score','bleu', 'chrf','meteor','rouge_we_1_f', 'rouge_we_2_f', 'rouge_we_3_f']
sorted_metrics = sorted(list(metrics))
articles = list(articles)

data = {}
output_data ={}
val_data ={}
max_coherence, max_consistency, max_fluency, max_relevance = [], [], [], []
for metric in sorted_metrics:
    if metric == "id":
        continue
    coherence_scores, consistency_scores, fluency_scores, relevance_scores = [], [], [], []
    coherence_scores_val, consistency_scores_val, fluency_scores_val, relevance_scores_val = [], [], [], []
    metric_scores = []
    metric_scores_val = []
    for summ_id in summ_ids:
        cur_metric = []
        cur_coherence, cur_consistency, cur_fluency, cur_relevance = [], [], [], []
        cur_metric_val = []
        cur_coherence_val, cur_consistency_val, cur_fluency_val, cur_relevance_val = [], [], [], []
        for article in articles:
            try:
                cur_metric.append(article2systemscores[article][summ_id][metric])
            except Exception as e:
                print(e)
                import pdb;pdb.set_trace()
            cur_coherence.append(article2humanscores[article][summ_id]["coherence"])
            cur_consistency.append(article2humanscores[article][summ_id]["consistency"])
            cur_fluency.append(article2humanscores[article][summ_id]["fluency"])
            cur_relevance.append(article2humanscores[article][summ_id]["relevance"])

        metric_scores.append(np.mean(cur_metric[:90]))

        coherence_scores.append(np.mean(cur_coherence[:90]))
        consistency_scores.append(np.mean(cur_consistency[:90]))
        fluency_scores.append(np.mean(cur_fluency[:90]))
        relevance_scores.append(np.mean(cur_relevance[:90]))

        metric_scores_val.append(np.mean(cur_metric[90:]))
    
        coherence_scores_val.append(np.mean(cur_coherence[90:]))
        consistency_scores_val.append(np.mean(cur_consistency[90:]))
        fluency_scores_val.append(np.mean(cur_fluency[90:]))
        relevance_scores_val.append(np.mean(cur_relevance[90:]))


    data[metric]=np.array(metric_scores)
    output_data['coherence_scores'] = np.array(coherence_scores)
    output_data['consistency_scores'] = np.array(consistency_scores)
    output_data['fluency_scores'] = np.array(fluency_scores)
    output_data['relevance_scores'] = np.array(relevance_scores)

    val_data[metric]=np.array(metric_scores_val)
    val_data['coherence_scores'] = np.array(coherence_scores_val)
    val_data['consistency_scores'] = np.array(consistency_scores_val)
    val_data['fluency_scores'] = np.array(fluency_scores_val)
    val_data['relevance_scores'] = np.array(relevance_scores_val)


   


import pandas as pd

df = pd.DataFrame({'rouge1': data['rouge_1_f_score'],'rouge2': data['rouge_2_f_score'],'rouge3': data['rouge_3_f_score'],'rouge4': data['rouge_4_f_score'],'rougeL': data['rouge_l_f_score'],'bleu': data['bleu'], 'chrf': data['chrf'],'meteor':data['meteor'],'rouge_we_1_f': data['rouge_we_1_f'], 'rouge_we_2_f': data['rouge_we_2_f'], 'rouge_we_3_f': data['rouge_we_3_f'],'Coherence': output_data['coherence_scores'],'Consistency': output_data['consistency_scores'],'Fluency': output_data['fluency_scores'],'Relevance': output_data['relevance_scores']})

val_df = pd.DataFrame({'rouge1': val_data['rouge_1_f_score'],'rouge2': val_data['rouge_2_f_score'],'rouge3': val_data['rouge_3_f_score'],'rouge4': val_data['rouge_4_f_score'],'rougeL': val_data['rouge_l_f_score'],'bleu': val_data['bleu'], 'chrf': val_data['chrf'],'meteor':val_data['meteor'],'rouge_we_1_f': val_data['rouge_we_1_f'],'rouge_we_2_f': val_data['rouge_we_2_f'],'rouge_we_3_f': val_data['rouge_we_3_f'],'Coherence': val_data['coherence_scores'],'Consistency': val_data['consistency_scores'],'Fluency': val_data['fluency_scores'],'Relevance': val_data['relevance_scores']})






from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
regr = MultiOutputRegressor(RandomForestRegressor()).fit(df[['rouge1','rouge2','rouge3','rouge4','rougeL','bleu', 'chrf','meteor','rouge_we_1_f', 'rouge_we_2_f', 'rouge_we_3_f']], df[['Coherence','Consistency','Fluency','Relevance']])

y_test = val_df[['Coherence','Consistency','Fluency','Relevance']]
x_test = val_df[['rouge1','rouge2','rouge3','rouge4','rougeL','bleu', 'chrf','meteor','rouge_we_1_f', 'rouge_we_2_f', 'rouge_we_3_f']]
import csv

#print(model_data)




mse = mean_squared_error(y_test, regr.predict(x_test))
print('mse is '+str(mse))
predict_data = pd.read_csv('rouge_metrics.csv')  
model_data = predict_data[['rouge1','rouge2','rouge3','rouge4','rougeL','bleu', 'chrf','meteor','rouge_we_1_f', 'rouge_we_2_f', 'rouge_we_3_f']]
summ_data = predict_data[['index','gold','generated']]
summ_values = summ_data.values.tolist()
rouge_values = model_data.values.tolist()
predicted_evals = regr.predict(model_data)
a = np.array(predicted_evals)
print('human metrics dataset level')
print(a.mean(axis = 0))
print()

b = np.array(model_data)
print('automatic dataset level')
print(b.mean(axis = 0))
print()
#print(predicted_evals)
for estimator in regr.estimators_:
    #print(decision_path(model_data))
    weights = pd.DataFrame(estimator.feature_importances_, model_data.columns, columns=['Coefficients'])
    print(weights)
f = 'predicted_scores.csv'
fields = ['index','gold','generated','Rouge1','Rouge2','Rouge3','Rouge4','RougeL','bleu', 'chrf','meteor','rouge_we_1_f', 'rouge_we_2_f', 'rouge_we_3_f','Coherence','Consistency','Fluency','Relevance']
with open(f, 'w') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    for i in range(6400):
      csvwriter.writerow(summ_values[i]+rouge_values[i]+list(predicted_evals[i]))
