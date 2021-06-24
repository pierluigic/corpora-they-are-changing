import argparse
import os
import numpy as np
from scipy.stats import  rankdata, spearmanr

def normalize(m):
    matrix=np.array(m)
    if len(matrix.shape)>1:
       norm=np.linalg.norm(matrix,axis=1)
       norm[norm==0]=1
       matrix= matrix/norm[:,np.newaxis]
    else:
       norm=np.linalg.norm(matrix)
       if not norm==0:
          matrix= matrix/norm
    return matrix

def compute_sim(path,name):

    if not os.path.exists("sims"):
        os.mkdir("sims")

    V = []
    M = [[],[]]

    for j,fname in enumerate(os.listdir(path)):

        with open("%s/%s"%(path,fname),'r') as f:
            for line in f:
                    line = line.split()
                    w = line[0]
                    vec = [float(v) for v in line[1:]]
                    if j==0:
                        V.append(w)
                    M[j].append(np.array(vec))

        M[j] = normalize(np.array(M[j]))

    s = []

    with open("%s/%s.txt" % ("sims", name), 'w+') as f:
        for j in range(len(V)):
                 sim = np.dot(M[0][j],M[1][j])
                 f.write("%s\t%f\n"%(V[j],sim))
                 s.append(sim)

    return V,s


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("name_first", help="First corpus name")
    parser.add_argument("path_first", help="First corpus vectors path")
    parser.add_argument("name_second", help="Second corpus name")
    parser.add_argument("path_second", help="Second corpus vectors path")
    args = parser.parse_args()


    V0,s0 = compute_sim(args.path_first,args.name_first)
    V1,s1 = compute_sim(args.path_second,args.name_second)

    V = set(V0).intersection(V1)

    rev_V0 = {w:j for j,w in enumerate(V0)}
    rev_V1 = {w:j for j,w in enumerate(V1)}

    new_s0 = []
    new_s1 = []

    for w in V:
        new_s0.append(s0[rev_V0[w]])
        new_s1.append(s1[rev_V1[w]])

    correlation = spearmanr(rankdata(new_s0),rankdata(new_s1))

    print(correlation)




