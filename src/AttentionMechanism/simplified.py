import torch
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

query = inputs[1] #query 2
attn_scores_2 = torch.empty(inputs.shape[0])

for i, x_i in enumerate(inputs):
    # we use dot product as it's a fast way to measure how aligned/similar two vectors are with each other
    # dot product q * k is large when vectors point in a similar direction, small/negative when they differ
    attn_scores_2[i] = torch.dot(x_i, query)

print(attn_scores_2)

# normalizing
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print(f"Attention weights: {attn_weights_2_tmp}")
print(f"Sum: {attn_weights_2_tmp.sum()}")

# it's better to normalize using softmax as it's better at managing extreme values and offers more favorable gradient
# properties
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)
attn_weights_2_naive = softmax_naive(attn_scores_2)
print(f"Attention weights: {attn_weights_2_naive}")
print(f"Sum: {attn_weights_2_naive.sum()}")

#pytorch softmax version
attn_weights_2 = torch.softmax(attn_scores_2, dim=-1)
print(f"Attention weights: {attn_weights_2}")
print(f"Sum: {attn_weights_2.sum()}")


# now we can calculate the context vector
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i
print(context_vec_2)


# attn_scores for each word compared to each word.
"""attn_scores = torch.empty(6,6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)

print(attn_scores)
"""
# for better performance we can use matrix multiplication
attn_scores = inputs @ inputs.T
print(attn_scores)
# and normalize
attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)
# ensure that the rows = 1
row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print(f"Row 2 sum: {row_2_sum}")
print(f"All row sums: {attn_weights.sum(dim=-1)}")

# finally we compute all the context vectors via matrix multiplication
all_context_vecs = attn_weights @ inputs
print(all_context_vecs)
