import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

import pyswarms as ps


class PSO:

    MODEL_SVR = 'svr'
    MODEL_KNN = 'knn'
    MODEL_DECISION_TREE = 'decision_tree'
    MODEL_RF = 'random_forest'
    MODEL_GBRT = 'GBRT'
    MODEL_LSTM = 'lstm'

    SUPPORTED_MODELS = [MODEL_SVR, MODEL_KNN, MODEL_DECISION_TREE, MODEL_RF, MODEL_GBRT]

    def __init__(self, c1=0.5, c2=0.5, w=0.9, n_particles=10, model_name='', seed=1):
        self.options = {'c1': c1, 'c2': c2, 'w': w}
        self.n_particles = n_particles
        self.model_name = model_name
        self.best_pos = None
        self.params, self.bounds = self.get_params_and_bounds(model_name)
        dims = len(self.params)
        self.optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=dims,
                                                 options=self.options, bounds=self.bounds)
        # 创建模型祥光
        self.seed = seed

    def get_params_and_bounds(self, model_name):
        params = []
        bounds = ()
        if model_name == self.MODEL_RF:
            params = ['n_estimators', 'max_depth']
            # bounds = ([100, 10, 2, 1], [300, 100, 5, 3])
            bounds = ([50, 2], [200, 20])
        elif model_name == self.MODEL_SVR:
            params = ['C']
            bounds = ([0.01], [100])
        elif model_name == self.MODEL_KNN:
            params = ['n_neighbors']
            bounds = ([2], [10])
        elif model_name == self.MODEL_DECISION_TREE:
            params = ['min_samples_split', 'min_samples_leaf']
            bounds = ([2, 1], [5, 5])
        elif model_name == self.MODEL_GBRT:
            params = ['n_estimators', 'learning_rate', 'max_depth']
            bounds = ([50, 0.001, 2], [200, 0.1, 20])

        return params, bounds

    def get_regressor(self, param, particle_idx):
        if particle_idx >= self.n_particles:
            print('get_regressor... Invalid particle index.')
            return None
        _reg = None
        if self.model_name == self.MODEL_RF:
            n_estimators = int(param[particle_idx][0])
            max_depth = int(param[particle_idx][1])
            return RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=self.seed)
        elif self.model_name == self.MODEL_SVR:
            C = param[particle_idx][0]
            return SVR(C=C)
        elif self.model_name == self.MODEL_KNN:
            n_estimators = int(param[particle_idx][0])
            return KNeighborsRegressor(n_neighbors=n_estimators)
        elif self.model_name == self.MODEL_DECISION_TREE:
            min_samples_split = int(param[particle_idx][0])
            min_samples_leaf = int(param[particle_idx][1])
            return DecisionTreeRegressor(min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                         random_state=self.seed)
        elif self.model_name == self.MODEL_GBRT:
            n_estimators = int(param[particle_idx][0])
            learning_rate = param[particle_idx][1]
            max_depth = int(param[particle_idx][2])
            return GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate,
                                             max_depth=max_depth, random_state=self.seed)
        return _reg

    def run(self, x_train, y_train, x_test, y_test):

        def objective(param):
            scores = np.zeros(self.n_particles)
            for i in range(self.n_particles):
                _reg = self.get_regressor(param, i)
                _reg.fit(x_train, y_train)
                _y_pred = _reg.predict(x_test)
                scores[i] = metrics.mean_absolute_error(y_test, _y_pred)

                # est.append(i_n_estimators)
                # dep.append(i_max_depth)
                # mae.append(metrics.mean_absolute_error(y_test, _y_pred))
            return np.mean(scores)

        best_cost, self.best_pos = self.optimizer.optimize(objective, iters=25)

        # print('best cost: '.format(best_cost))
        # print('best pos: '.format(self.best_pos))
        print('pso... Info for the [{}] optimization result below:'.format(self.model_name))
        # reg = RandomForestRegressor(n_estimators=int(best_pos[0]), max_depth=int(best_pos[1]))
        # reg.fit(x_train, y_train)
        # y_pred = reg.predict(x_test)
        # return self.params, self.best_pos
        return self.get_best_regressor()

    def get_best_regressor(self):
        param = self.best_pos
        if self.model_name == self.MODEL_RF:
            n_estimators = int(param[0])
            max_depth = int(param[1])
            return RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=self.seed)
        elif self.model_name == self.MODEL_SVR:
            C = param[0]
            return SVR(C=C)
        elif self.model_name == self.MODEL_KNN:
            n_estimators = int(param[0])
            return KNeighborsRegressor(n_neighbors=n_estimators)
        elif self.model_name == self.MODEL_DECISION_TREE:
            min_samples_split = int(param[0])
            min_samples_leaf = int(param[1])
            return DecisionTreeRegressor(min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                         random_state=self.seed)
        elif self.model_name == self.MODEL_GBRT:
            n_estimators = int(param[0])
            learning_rate = param[1]
            max_depth = int(param[2])
            return GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate,
                                             max_depth=max_depth, random_state=self.seed)
        return None
