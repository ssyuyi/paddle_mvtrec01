from sklearn.covariance import LedoitWolf
import paddle
import pickle


class GaussianDensityPaddle(object):
    """Gaussian Density estimation similar to the implementation used by Ripple et al.
    The code of Ripple et al. can be found here: https://github.com/ORippler/gaussian-ad-mvtec.
    """
    def __init__(self):
        super(GaussianDensityPaddle, self).__init__()
        self.inv_cov = None
        self.best_threshold = None

    def fit(self, embeddings):
        self.mean = paddle.mean(embeddings, axis=0)
        self.inv_cov = paddle.to_tensor(LedoitWolf().fit(embeddings.cpu()).precision_,dtype="float32")

    def load(self, file):
        with open(file,"rb") as f:
            info = pickle.load(f)
            self.inv_cov = paddle.to_tensor(info['inv_cov'])
            self.mean = paddle.to_tensor(info['mean'])
            self.best_threshold = paddle.to_tensor(info['best_threshold'])

    def predict(self, embeddings):
        scores = self.mahalanobis_distance(embeddings, self.mean, self.inv_cov)
        # invert scores, so they fit to the class labels for the auc calculation
        return scores

    @staticmethod
    def mahalanobis_distance(
            values, mean, inv_covariance
    ):
        """Compute the batched mahalanobis distance.
        values is a batch of feature vectors.
        mean is either the mean of the distribution to compare, or a second
        batch of feature vectors.
        inv_covariance is the inverse covariance of the target distribution.

        from https://github.com/ORippler/gaussian-ad-mvtec/blob/4e85fb5224eee13e8643b684c8ef15ab7d5d016e/src/gaussian/model.py#L308
        """

        if mean.dim() == 1:  # Distribution mean.
            mean = mean.unsqueeze(0)
        x_mu = values - mean  # batch x features
        dist = paddle.einsum("im,mn,in->i", x_mu, inv_covariance, x_mu)

        return paddle.sqrt(dist)
