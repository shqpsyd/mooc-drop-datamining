from sklearn.cluster import KMeans
import numpy as np
X = np.array([30,1,40,2,50]).reshape(-1,1)
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
p0 = 0
p1 = 0
c0 = 0
c1 = 0
for i in range(5):
	if kmeans.labels_[i] == 0:
		p0 += X[i]
		c0+=1
	else:
		p1 += X[i]
		c1+=1
p = 0
if c1>c0:
	p = p1/float(c1)
else:
	p = p0/c0
print p[0]



#kmeans.cluster_centers_
