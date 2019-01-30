



class LocalityPreservingProjections(BaseEstimator, TransformerMixin):
    """ Scikit-Learn compatible class for Locality Preserving Projections

    Parameters
    ----------

    n_components : integer, optional, default=2
        number of features for the manifold (=< features of data)

    eig_solver : string ['dense', 'multi', 'sparse'], optional, default='dense'
        eigenvalue solver method

    norm_lap : bool, optional, default=False
        normalized laplacian or not

    tol : float, optional, default=1E-12
        stopping criterion for eigenvalue decomposition of the Laplacian matrix
        when using arpack or multi

    normalization : string ['degree', 'identity'], default = None ('degree')
        normalization parameter for eigenvalue problem

    n_neighbors :

    Attributes
    ----------

    _spectral_embedding :

    _embedding_tuner :


    References
    ----------

    Original Paper:
        http://www.cad.zju.edu.cn/home/xiaofeihe/LPP.html
    Inspired by Dr. Jake Vanderplas' Implementation:
        https://github.com/jakevdp/lpproj

    """
    def __init__(self, n_components=2, eig_solver = 'dense', norm_laplace = False,
                 eigen_tol = 1E-12, regularizer = None,
                 normalization = None, n_neighbors = 2,neighbors_algorithm = 'brute',
                 metric = 'euclidean',n_jobs = 1,weight = 'heat',affinity = None,
                 gamma = 1.0,trees = 10,sparse = True,random_state = 0):
        self.n_components = n_components
        self.eig_solver = eig_solver
        self.regularizer = regularizer
        self.norm_laplace = norm_laplace
        self.eigen_tol = eigen_tol
        self.normalization = normalization
        self.n_neighbors = n_neighbors
        self.neighbors_algorithm = neighbors_algorithm
        self.metric = metric
        self.n_jobs = n_jobs
        self.weight = weight
        self.affinity = affinity
        self.gamma = gamma
        self.trees = trees
        self.sparse = False,
        self.random_state = random_state

    def fit(self, X, y=None):

        # TODO: handle sparse case of data entry
        # check the array
        X = check_array(X)

        # compute the adjacency matrix for X
        W = compute_adjacency(X,
                              n_neighbors=self.n_neighbors,
                              weight=self.weight,
                              affinity=self.affinity,
                              metric=self.metric,
                              neighbors_algorithm=self.neighbors_algorithm,
                              gamma=self.gamma,
                              trees=self.trees,
                              n_jobs=self.n_jobs)

        # compute the projections into the new space
        self.eigVals, self.projection_ = self._spectral_embedding(X, W)

        return self

    def transform(self, X):

        # check the array and see if it satisfies the requirements
        X = check_array(X)
        if self.sparse:
            return X.dot(self.projection_)
        else:
            return np.dot(X, self.projection_)

    def _spectral_embedding(self, X, W):

        # find the eigenvalues and eigenvectors
        return linear_graph_embedding(adjacency=W, data=X,
                                      norm_laplace=self.norm_laplace,
                                      normalization=self.normalization,
                                      eig_solver=self.eig_solver,
                                      eigen_tol=self.eigen_tol)

