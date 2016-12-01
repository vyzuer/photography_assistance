print(__doc__)

import numpy as np

from sklearn.ensemble import ExtraTreesClassifier

f_file = "./DB/fv.list"
v_file = "./DB/ev.score"

# load features
X = np.loadtxt(f_file)
target = np.loadtxt(v_file)

y = np.matrix.round(target)

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=500,
                              random_state=0)

forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(13):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
import pylab as pl
pl.figure()
pl.title("Feature importances")
pl.bar(range(13), importances[indices],
       color="b", yerr=std[indices], align="center")
pl.xticks(range(13), indices)
pl.xlim([-1, 13])
pl.show()

