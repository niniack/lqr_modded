"""
Wrappers for transformer blocks. 
From https://github.com/trustworthyrobotics/lqr-activation-steering
"""

def tf_block_wrapper(block, attention_mask, position_ids, position_embeddings, x):
    """
    Block wrapper for transformer layers.
    """
    output = block(x.unsqueeze(0), attention_mask=attention_mask,
                   position_ids=position_ids, position_embeddings=position_embeddings)
    return output[0] if isinstance(output, (tuple, list)) else output

def transformerBlockControl(tf, x, u):
    """
    Control wrapper for transformer blocks.
    Adds control input `u` to the last token of the output.
    """
    x_next = tf(x)
    x_next[..., -1, :] = x_next[..., -1, :] + u
    return x_next