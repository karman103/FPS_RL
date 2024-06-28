import torch.nn as nn
import torch

class BucketedEmbedding(nn.Embedding):

    def __init__(self, bucket_size, num_embeddings, *args, **kwargs):
        self.bucket_size = bucket_size
        real_num_embeddings = (num_embeddings + bucket_size - 1) // bucket_size
        super(BucketedEmbedding, self).__init__(real_num_embeddings, *args, **kwargs)

    def forward(self, indices):
        if indices.dtype != torch.long:
            indices = indices.long()
        # Perform integer division to get integer indices
        bucket_indices = indices // self.bucket_size
        return super(BucketedEmbedding, self).forward(bucket_indices)
