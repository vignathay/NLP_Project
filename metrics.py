

import nltk
from nltk.translate.meteor_score import meteor_score, single_meteor_score
from nltk import word_tokenize
nltk.download('punkt')
nltk.download('wordnet')
import numpy as np

def compute_meteor(  prediction, reference, alpha=0.9, beta=3, gamma=0.5):
    score =  meteor_score([word_tokenize(reference[0])], word_tokenize(prediction[0]), alpha=alpha, beta=beta, gamma=gamma) 
    return score

def compute_bleu(  prediction, reference, sent_smooth_method='exp', sent_smooth_value=None, sent_use_effective_order=True, \
       smooth_method='exp', smooth_value=None, force=False, lowercase=False, \
       use_effective_order=False, n_workers=24):
    score = sacrebleu.sentence_bleu(prediction[0], reference, smooth_method=sent_smooth_method, smooth_value=sent_smooth_value, use_effective_order=sent_use_effective_order)
    return score.score

from datasets import load_metric
rouge = load_metric("rouge")
def compute_rouge_metrics(pred_str, actual_str, metric_name):
  rouge_output = rouge.compute(
      predictions=pred_str, references=actual_str, rouge_types=[metric_name]
  )[metric_name].mid

  return round(rouge_output.fmeasure, 4)
      #return {
          #"rouge2_precision": round(rouge_output.precision, 4),
          #"rouge2_recall": round(rouge_output.recall, 4),
         # "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
     # }

import sacrebleu
def compute_chrf_score(summary, reference):
  if not isinstance(reference, list):
    reference = [reference]
  score = sacrebleu.sentence_chrf(summary[0], reference, char_order=6, word_order=0, beta=2, remove_whitespace=True)
  return score.score
  return score_dict

import csv
all_rouge_metrics = ['rouge1','rouge2','rouge3','rouge4','rougeL']
metrics = ['rouge1','rouge2','rouge3','rouge4','rougeL',  'bleu', 'chrf', 'meteor']
file = open('LED_total_data.csv')
csvreader = csv.reader(file)
actuals =[]
preds = []
count=0
#gold','genrated','rouge1','rouge2','rouge3','rouge4','rougeL','bleu', 'chrf','meteor'
with open('All_Computed_Metrics', 'w') as csvfile: 
  csvwriter = csv.writer(csvfile)
  csvwriter.writerow(['index','gold','genrated','rouge1','rouge2','rouge3','rouge4','rougeL','bleu', 'chrf','meteor'])
  for row in csvreader:
    count += 1
    if count == 1:
      continue
    values = [row[0], row[2], row[1].strip('\n') ]
    print(count)
    for metric in metrics:
      val = compute_metrics([row[1].strip('\n')], [row[2]], metric)
      values.append(val)
    csvwriter.writerow(values)

from datasets import load_metric
rouge = load_metric("rouge")
def compute_metrics(pred_str, actual_str, metric_name):
    if metric_name in all_rouge_metrics:
      return compute_rouge_metrics(pred_str, actual_str, metric_name)
    elif metric_name  == 'bleu':
      return compute_bleu(pred_str, actual_str)
    elif metric_name == 'chrf':
      return compute_chrf_score(pred_str, actual_str)
    elif metric_name == 'meteor':
      return compute_meteor(pred_str, actual_str)

actual_str = "Two cars loaded with gasoline and nails found abandoned in London Friday . 52 people killed on July 7, 2005 after bombs exploded on London bus, trains . British capital wracked by violence by the IRA for years"
pred_str = "Five cars loaded with gasoline and nails found abandoned in London Friday . 99 people killed on Septmeber 19, 1990 after bombs exploded on London bus, trains . British capital wracked by violence by the IRA for years"
nltk.download('omw-1.4')
all_rouge_metrics = ['rouge1','rouge2','rouge3','rouge4','rougeL']
metrics = ['rouge1','rouge2','rouge3','rouge4','rougeL',  'bleu', 'chrf', 'meteor']
for metric in metrics:
  print(compute_metrics([pred_str], [actual_str], metric))