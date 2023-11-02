from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from IPython.display import Image  
import pydotplus
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # konfiguruje generator znaczników i mapę kolorów
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # rysuje wykres powierzchni decyzyjnej
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # rysuje wykres wszystkich próbek
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl, edgecolor='black')


iris = datasets.load_iris()
X = iris.data[:, [1, 2]]
y = iris.target

# Zadanie 1: Rozdziel zestaw danych na podzbiory uczący i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# # Zadanie 2: Sprawdź działanie drzewa dla entropii i współczynnika Giniego - porównaj wyniki i uargumentuj rezultaty
clf_ent = DecisionTreeClassifier(criterion='entropy', random_state=1)
clf_gini = DecisionTreeClassifier(criterion='gini', random_state=1)

clf_ent.fit(X_train, y_train)
clf_gini.fit(X_train, y_train)


plot_decision_regions(X=X, y=y, classifier=clf_ent)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.legend(loc='upper left')
plt.show()

plot_decision_regions(X=X, y=y, classifier=clf_gini)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.legend(loc='upper left')
plt.show()

tree.plot_tree(clf_ent) 
plt.title('Decision Tree - Entropy')
plt.show()

tree.plot_tree(clf_gini) 
plt.title('Decision Tree - Gini')
plt.show()

# Export decision tree graphs to PNG
# dat_ent = export_graphviz(clf_ent, out_file=None,  
#                 filled=True, rounded=True,
#                 special_characters=True)
# graph_ent = pydotplus.graph_from_dot_data(dat_ent)  
# graph_ent.write_png('decision_tree_entropy.png')
# Image(graph_ent.create_png())

# dat_gini = export_graphviz(clf_gini, out_file=None,  
#                 filled=True, rounded=True,
#                 special_characters=True)
# graph_gini = pydotplus.graph_from_dot_data(dat_gini)  
# graph_gini.write_png('decision_tree_gini.png')
# Image(graph_gini.create_png())

####################################################3

# Zadanie 3: Sprawdź działanie drzewa dla różnych głębokości drzewa - porównaj wyniki i uargumentuj rezultaty
max_depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
train_accs = []
test_accs = []

for depth in max_depths:
    clf = DecisionTreeClassifier(criterion='entropy', random_state=1, max_depth=depth)
    clf.fit(X_train, y_train)
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

plt.plot(max_depths, train_accs, label='Training accuracy')
plt.plot(max_depths, test_accs, label='Test accuracy')
plt.xlabel('Max depth')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#####################################


#Zadanie 4: Sprawdź działanie lasów losowych dla różnej liczby drzew decyzyjnych - porównaj wyniki i uargumentuj rezultaty
n_trees = [1, 5, 10, 20, 50, 100, 200]
train_accs = []
test_accs = []

for n in n_trees:
    clf = RandomForestClassifier(n_estimators=n, criterion='entropy', random_state=1)
    clf.fit(X_train, y_train)
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

plt.plot(n_trees, train_accs, label='Training accuracy')
plt.plot(n_trees, test_accs, label='Test accuracy')
plt.xlabel('Number of trees')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

