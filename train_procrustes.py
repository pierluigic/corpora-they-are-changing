import multiprocessing
import os
from time import time
import argparse
from gensim.models import Word2Vec
from scipy.spatial import procrustes
import numpy as np


def get_sentences(ylist):
      class MySentences(object):
           def __init__(self,ylist):
               self.ylist=ylist
           def __iter__(self):
               for y in self.ylist:
                   with open(y,'r') as f:
                       for line in f:
                           words=line.lower().split()
                           yield words
      return MySentences(ylist)



class W2V():
      def __init__(self,fname,args):

          self.min_count=args.min_count
          self.window=args.window
          self.size=args.size
          self.sample=args.sample
          self.negative=args.negative
          self.epochs=args.epochs
          self.alpha=args.alpha
          self.sentences=get_sentences([fname])
      
      def train(self):
          cores = multiprocessing.cpu_count() # Coudnt the number of cores in a computer
              
          w2v_model = Word2Vec(min_count=self.min_count,
                     window=self.window,
                     size=self.size,
                     sg=1,
                     sample=self.sample,  
                     negative=self.negative,
                     alpha=self.alpha,
                     workers=cores-1)
          t = time()

          w2v_model.build_vocab(self.sentences, progress_per=10000)
          print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))
          t = time()

          w2v_model.train(self.sentences, total_examples=w2v_model.corpus_count, epochs=self.epochs, report_delay=1)

          print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
          self.w2v_model=w2v_model




def align(collection):
    embs = []

    for model in collection:
        embs.append(model)

    for k in range(1,len(embs)):
        embs[k-1],embs[k],disp = procrustes(embs[k-1],embs[k])

        embs=np.array(embs)

    return embs

def train_procrustes(args):

      fnames = []

      if not os.path.exists("%s_%s"%(args.corpus_name,"vectors")):
         os.mkdir("%s_%s"%(args.corpus_name,"vectors"))

      V=set()
      models=[]
      for fname in os.listdir(args.path):
          fnames.append(fname)
          w2v=W2V("%s/%s"%(args.path,fname),args)
          w2v.train()
          V_temp=set()
          models.append(w2v.w2v_model)
          for k in w2v.w2v_model.wv.vocab.keys():
              V_temp.add(k)
          if len(V)>0:
             V=V.intersection(V_temp)
          else:
             V=V_temp

      V=list(V)
      c=[]
      collection=[]
      for i in range(len(models)):
          for w in V:
              c.append(models[i].wv[w])
          collection.append(c)
          c=[]
      collection=np.array(collection)
      embs=align(collection)

      for j in range(len(embs)):
          with open('%s_vectors/%s_vectors.txt'%(args.corpus_name,fnames[j]),'w+') as f:
                for i in range(len(V)):
                    w=V[i]
                    f.write(w+' '+' '.join([str(v) for v in embs[j][i]])+'\n')



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Corpora path")
    parser.add_argument("corpus_name", help="Project name")
    parser.add_argument("--min_count", help="Ignores all words with total frequency lower than this.", default=10)
    parser.add_argument("--window", help="Maximum distance between the current and predicted word within a sentence.", default=5)
    parser.add_argument("--size", help="Dimensionality of the word vectors.", default=300)
    parser.add_argument("--sample", help="The threshold for configuring which higher-frequency words are randomly downsampled, useful range is (0, 1e-5).", default=0.001)
    parser.add_argument("--alpha", help="The initial learning rate.", default=0.025)
    parser.add_argument("--negative", help="If > 0, negative sampling will be used, the int for negative specifies how many noise words should be drawn (usually between 5-20). If set to 0, no negative sampling is used.", default=5)
    parser.add_argument("--epochs", help="Number of iterations (epochs) over the corpus", default=5)
    args = parser.parse_args()

    train_procrustes(args)
