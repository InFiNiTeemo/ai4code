import torch
import numpy as np
def _generate_relative_positions_matrix(length, max_relative_position,
                                        cache=False):
    """Generates matrix of relative positions between inputs."""
    if not cache:
        range_vec = torch.arange(length)
        range_mat = range_vec.repeat(length).view(length, length)
        distance_mat = range_mat - torch.t(range_mat)
    else:
        distance_mat = torch.arange(-length + 1, 1, 1).unsqueeze(0)

    distance_mat_clipped = torch.clamp(distance_mat, -max_relative_position, max_relative_position)
    final_mat = distance_mat_clipped + max_relative_position

    return final_mat


def _generate_relative_positions_embeddings(length, depth, max_relative_position=127):
    vocab_size = max_relative_position * 2 + 1
    range_vec = torch.arange(length)
    range_mat = range_vec.repeat(length).view(length, length)
    distance_mat = range_mat - torch.t(range_mat)
    distance_mat_clipped = torch.clamp(distance_mat, -max_relative_position, max_relative_position)
    final_mat = distance_mat_clipped + max_relative_position
    embeddings_table = np.zeros([vocab_size, depth])

    for pos in range(vocab_size):
        for i in range(depth // 2):
            embeddings_table[pos, 2 * i] = np.sin(pos / np.power(10000, 2 * i / depth))
            embeddings_table[pos, 2 * i + 1] = np.cos(pos / np.power(10000, 2 * i / depth))

    embeddings_table_tensor = torch.tensor(embeddings_table).float()
    flat_relative_positions_matrix = final_mat.view(-1)
    one_hot_relative_positions_matrix = torch.nn.functional.one_hot(flat_relative_positions_matrix,
                                                                    num_classes=vocab_size).float()
    embeddings = torch.matmul(one_hot_relative_positions_matrix, embeddings_table_tensor)
    my_shape = list(final_mat.size())
    my_shape.append(depth)
    embeddings = embeddings.view(my_shape)
    return embeddings
