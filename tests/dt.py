from sklearn.datasets import load_boston
from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeRegressor

boston = load_boston()
regressor = DecisionTreeRegressor(random_state=0)
print boston.data.shape
print boston.target.shape
print cross_val_score(regressor, boston.data, boston.target, cv=10)
