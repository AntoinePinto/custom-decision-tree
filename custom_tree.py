from functools import partial

import numpy as np

from tqdm import tqdm


def get_metric(metric_vars):

    y = metric_vars[:, 0]
    odds = metric_vars[:, 1]

    metric = np.sum((np.mean(y) > (1 / odds)) * (odds * y - 1))

    return metric

def get_delta(split, metric_vars, optimisation_method='minimisation'):

    delta =  get_metric(metric_vars) - (get_metric(metric_vars[split]) + get_metric(metric_vars[np.invert(split)]))

    if optimisation_method == 'maximisation':
        delta = - delta

    return delta

def get_best_split_from_var(var, metric_vars, optimisation_method='minimisation'):

    splits = np.sort(np.unique(var))[:-1]

    if len(splits) == 0:
        return np.nan, np.nan

    if len(splits) > 200:
        splits = np.quantile(splits, [i/200 for i in range(200)])

    deltas = np.array(list(map(partial(get_delta, metric_vars=metric_vars, optimisation_method=optimisation_method), [var > split for split in splits])))

    optimum_value = deltas[np.argmax(deltas)]
    best_split = splits[np.argmax(deltas)]

    return best_split, optimum_value

class CustomDecisionTreeClassifier:

    def __init__(self, max_depth=5):

        self.max_depth = max_depth

    def fit(self, X, y, metric_vars, optimisation_method='minimisation'):

        metric_vars = np.array(metric_vars)
        y = np.array(y)
        X = np.array(X)
        id_vars = range(X.shape[1])

        partitions = {1: {'type_partition': 'leaf',
                          'depth': 0, 
                          'mask': np.repeat(True, len(y)),
                          'metric': get_metric(metric_vars),
                          'repartition': [np.sum(y == 0), np.sum(y == 1)]}}
        splitting = {}

        for depth in range(self.max_depth):

            for id_partition in list(partitions):

                part = partitions[id_partition]

                if (part['depth'] != depth) | (np.sum(part['repartition']) == 1):
                    continue

                mask = part['mask']

                best_split_by_var = list(map(partial(get_best_split_from_var, metric_vars=metric_vars[mask], optimisation_method=optimisation_method), [X[mask, i] for i in id_vars]))
                best_splits, optimum_values = [i for i, j in best_split_by_var], [j for i, j in best_split_by_var]

                if np.mean(np.isnan(optimum_values)) == 1:
                    continue

                id_optimum = np.nanargmax(optimum_values)
                id_var, split_value, optimum_value = id_vars[id_optimum], best_splits[id_optimum],  optimum_values[id_optimum]
                
                splitting[id_partition] = {'id_var': id_var, 'split_value': split_value, 'delta': optimum_value}

                mask_side1 = mask * (X[:, id_var] <= split_value)
                mask_side2 = mask * (X[:, id_var] > split_value)

                part['type_partition'] = 'branch'

                partitions[id_partition * 2] = {'type_partition': 'leaf',
                                                'depth': part['depth'] + 1,
                                                'mask': mask_side1, 
                                                'metric': get_metric(metric_vars[mask_side1]),
                                                'repartition': [np.sum(y[mask_side1] == 0), np.sum(y[mask_side1] == 1)]}

                partitions[id_partition * 2 + 1] = {'type_partition': 'leaf',
                                                    'depth': part['depth'] + 1,
                                                    'mask': mask_side2, 
                                                    'metric': get_metric(metric_vars[mask_side2]),
                                                    'repartition': [np.sum(y[mask_side2] == 0), np.sum(y[mask_side2] == 1)]}

                self.partitions = partitions
                self.splitting = splitting

    def predict_proba_x(self, x, return_exp_metric=False):

        id_partition = 1
        while True:

            if self.partitions[id_partition]['type_partition'] == 'leaf':
                rep = self.partitions[id_partition]['repartition']
                probas = [i/np.sum(rep) for i in rep]

                if return_exp_metric:
                    expected_metric = self.partitions[id_partition]['metric']
                    return probas, expected_metric

                return probas

            split = self.splitting[id_partition]

            if x[split['id_var']] <= split['split_value']:
                id_partition = id_partition * 2
            else:
                id_partition = id_partition * 2 + 1

    def predict_proba(self, X, return_exp_metric=False):

        X = np.array(X)

        probas = list(map(partial(self.predict_proba_x, return_exp_metric=return_exp_metric), [X[i,:] for i in range(len(X))]))

        return probas

    def print_tree(self, max_depth=1000, features_names=None, show_delta=True, show_metric=True, show_repartition=True):

        id_partition = 1
        while True:

            if (id_partition not in self.partitions):
                break

            part = self.partitions[id_partition]

            if part['depth'] <= max_depth:

                print("|         " * (1 + part['depth']) + '--- ', end="")
                print(f"node {id_partition}", end = "")

                if id_partition > 1:
                    id_first_parent = int((id_partition - (id_partition % 2 == 1)) / 2)
                    var = self.splitting[id_first_parent]['id_var']
                    var = features_names[var] if features_names is not None else f'feature {var}'
                    split_value = self.splitting[id_first_parent]['split_value']
                    split_type = '>' if id_partition % 2 == 1 else '<='
                    print(f" | {var} {split_type} {split_value}", end = '')
                    if show_delta is True:
                        delta = self.splitting[id_first_parent]['delta']
                        print(f" | Δ = {delta}", end = '')

                if show_metric is True:
                    metric = part['metric']
                    print(f" -> metric = {metric}", end="")
                if show_repartition is True:
                    print(f" | repartition = {part['repartition']}", end="")
                print('')

            if part['type_partition'] == 'branch':
                id_partition = id_partition * 2
            elif (id_partition % 2) == 0:
                id_partition = id_partition + 1
            else:
                parents, id = [], id_partition
                while id != 1:
                    id = int((id - (id % 2 == 1)) / 2)
                    parents.append(id)
                id_partition = next((id + 1 for id in parents if (id + 1 in self.partitions) and (id % 2 == 0)), None)

class CustomRandomForestClassifier:

    def __init__(self, n_estimators=100, max_depth=5):

        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def fit(self, X, y, metric_vars, optimisation_method='minimisation'):

        metric_vars = np.array(metric_vars)
        y = np.array(y)
        X = np.array(X)

        forest = {}
        for id_estimator in tqdm(range(self.n_estimators)):

            sub_var = np.random.choice(range(X.shape[1]), size=int(np.sqrt(X.shape[1])), replace=False)
            sub_obs = np.random.choice(range(X.shape[0]), size=X.shape[0], replace=True)
            
            model = CustomDecisionTreeClassifier(max_depth=self.max_depth)
            model.fit(X=X[sub_obs,:][:,sub_var], y=y[sub_obs], metric_vars=metric_vars[sub_obs], optimisation_method=optimisation_method)

            # Correct variable id that are incorectly encoded in model object due to variable sampling
            [d.update({'id_var': sub_var[d['id_var']]}) for d in model.splitting.values()]

            forest[id_estimator] = {'sub_obs': sub_obs, 'model': model}

        self.forest = forest

    def predict_proba_x(self, x, return_exp_metric=False):

        estimators_probas = []
        for id_estimator in self.forest:
            m = self.forest[id_estimator]['model']
            estimators_probas.append(m.predict_proba_x(x, return_exp_metric))

        if return_exp_metric is True:
            metric = np.mean(np.array([j for i, j in estimators_probas]))
            estimators_probas = np.array([i for i, j in estimators_probas])
            probas = list(np.mean(estimators_probas, axis=0))
            return probas, metric
        else:
            probas = list(np.mean(np.array(estimators_probas), axis=0))
            return probas

    def predict_proba(self, X, return_exp_metric=False):

        X = np.array(X)

        probas = list(map(partial(self.predict_proba_x, return_exp_metric=return_exp_metric), [X[i,:] for i in range(len(X))]))

        return probas