import torch

if __name__ == '__main__':
    feature_dim = 160
    n_output = 3
    bias = True
    no_scale_shift = False
    no_shift = True

    # suppose we have CH4
    c_vi = torch.rand([1, feature_dim])
    h_vi = torch.rand([1, feature_dim])
    # H embeddings in CH4 are the same
    vi = torch.cat([c_vi, h_vi, h_vi, h_vi, h_vi], dim=0)
    print(f"atom vector after last residual size: {vi.shape}")

    c_scale = torch.rand([1, 1])
    c_shift = torch.rand([1, 1])
    h_scale = torch.rand([1, 1])
    h_shift = torch.rand([1, 1])

    if no_scale_shift:
        c_scale.fill_(1.)
        c_shift.fill_(0.)
        h_scale.fill_(1.)
        h_shift.fill_(0.)
    elif no_shift:
        c_shift.fill_(0.)
        h_shift.fill_(0.)
    # The same atoms have the same scale and shift. Different atoms have different ones.
    scale_op = torch.cat([c_scale, h_scale, h_scale, h_scale, h_scale], dim=0)
    print(f"scale param size: {scale_op.shape}")
    shift_op = torch.cat([c_shift, h_shift, h_shift, h_shift, h_shift], dim=0)
    print(f"shift param size: {shift_op.shape}")

    lin_weight = torch.rand([feature_dim, n_output])
    if bias:
        lin_bias = torch.rand([1, n_output])
    else:
        lin_bias = 0.
    print(f"last linear param size: {lin_weight.shape}")

    print("********* results ********")

    # lin -> scale, shift -> sum
    tmp = torch.matmul(vi, lin_weight) + lin_bias
    tmp = scale_op * tmp + shift_op
    path1_result = torch.sum(tmp, dim=0, keepdim=True)
    print(f"lin -> scale, shift -> sum: {path1_result}")

    # scale, shift -> sum -> lin
    tmp = scale_op * vi + shift_op
    tmp = torch.sum(tmp, dim=0, keepdim=True)
    path2_result = torch.matmul(tmp, lin_weight) + lin_bias
    print(f"scale, shift -> sum -> lin: {path2_result}")
