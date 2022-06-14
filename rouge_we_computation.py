from summ_eval.metric import Metric
import nltk
nltk.download('stopwords')
import os
import requests
from multiprocessing import Pool
from collections import Counter
import gin
import bz2
from summ_eval.s3_utils import rouge_n_we, load_embeddings
from summ_eval.metric import Metric
dirname = os.path.dirname(os.path.abspath("__file__"))
if not os.path.exists(os.path.join(dirname, "embeddings")):
    os.mkdir(os.path.join(dirname, "embeddings"))
if not os.path.exists(os.path.join(dirname, "embeddings/deps.words")):
    print("Downloading the embeddings; this may take a while")
    url = "http://u.cs.biu.ac.il/~yogo/data/syntemb/deps.words.bz2"
    r = requests.get(url)
    d = bz2.decompress(r.content)
    with open(os.path.join(dirname, "embeddings/deps.words"), "wb") as outputf:
        outputf.write(d)

@gin.configurable
class RougeWeMetric(Metric):
    def __init__(self, emb_path=os.path.join(dirname, './embeddings/deps.words'), n_gram=3, \
                 n_workers=24, tokenize=True):
        self.word_embeddings = load_embeddings(emb_path)
        self.n_gram = n_gram
        self.n_workers = n_workers
        self.tokenize = tokenize

    def evaluate_example(self, summary, reference):
        if not isinstance(reference, list):
            reference = [reference]
        if not isinstance(summary, list):
            summary = [summary]
        score = rouge_n_we(summary, reference, self.word_embeddings, self.n_gram, \
                 return_all=True, tokenize=self.tokenize)
        score_dict = {f"rouge_we_{self.n_gram}_p": score[0], f"rouge_we_{self.n_gram}_r": score[1], \
                      f"rouge_we_{self.n_gram}_f": score[2]}
        return score_dict

    def evaluate_batch(self, summaries, references, aggregate=True):
        p = Pool(processes=self.n_workers)
        results = p.starmap(self.evaluate_example, zip(summaries, references))
        p.close()
        if aggregate:
            corpus_score_dict = Counter()
            for x in results:
                corpus_score_dict.update(x)
            for key in corpus_score_dict.keys():
                corpus_score_dict[key] /= float(len(summaries))
            return corpus_score_dict
        else:
            return results

    @property
    def supports_multi_ref(self):
        return True

rouge_we1 = RougeWeMetric(n_gram=1)
rouge_we2 = RougeWeMetric(n_gram=2)
rouge_we3 = RougeWeMetric(n_gram=3)

rouge_we1.evaluate_example(["Abstract_summary"],["Real_summary"])

import csv

with open("../../1500_2500_sheet.csv","r") as csvfile1,open("../../rouge_we_1500_2500.csv","w") as csvfile2:
  csvreader = csv.reader(csvfile1)
  csvwriter = csv.writer(csvfile2)
  csvwriter.writerow(['Rouge_we1','Rouge_we2','Rouge_we3'])
  i=0
  print("ncnnnnnnnnnn")
  print(csvreader)
  for row in csvreader:
    print("bvsmnv bmd")
    we1 = rouge_we1.evaluate_example([row[0]],[row[1]])['rouge_we_1_f']
    we2 = rouge_we2.evaluate_example([row[0]],[row[1]])['rouge_we_2_f']
    we3 = rouge_we3.evaluate_example([row[0]],[row[1]])['rouge_we_3_f']
    csvwriter.writerow([we1,we2,we3])
    print(i)
    print(row[0],row[1])
    i+=1
    if i>10:
      break

