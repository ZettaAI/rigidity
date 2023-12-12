import torch


def get_rigidity_map_zcxy(
    field: torch.Tensor, power: float = 2, diagonal_mult: float = 1.0
) -> torch.Tensor:
    # Kernel on Displacement field yields change of displacement

    if field.abs().sum() == 0:
        return torch.zeros(
            (field.shape[0], field.shape[2], field.shape[3]), device=field.device
        )

    batch = field.shape[0]
    diff_ker = torch.tensor(
        [
            [
                [[0, 0, 0], [-1, 1, 0], [0, 0, 0]],
                [[0, -1, 0], [0, 1, 0], [0, 0, 0]],
                [[-1, 0, 0], [0, 1, 0], [0, 0, 0]],
                [[0, 0, -1], [0, 1, 0], [0, 0, 0]],
            ]
        ],
        dtype=field.dtype,
        device=field.device,
    )

    diff_ker = diff_ker.permute(1, 0, 2, 3).repeat(2, 1, 1, 1)

    # Add distance between pixel to get absolute displacement
    diff_bias = torch.tensor(
        [1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 1.0, 1.0],
        dtype=field.dtype,
        device=field.device,
    )
    delta = torch.conv2d(field, diff_ker, diff_bias, groups=2, padding=[2, 2])
    # delta1 = delta.reshape(2, 4, *delta.shape[-2:]).permute(1, 2, 3, 0) # original
    delta = delta.reshape(batch, 2, 4, *delta.shape[-2:]).permute(0, 2, 3, 4, 1)

    # spring_lengths1 = torch.norm(delta1, dim=3)
    spring_lengths = torch.norm(delta, dim=-1)

    spring_defs = torch.stack(
        [
            spring_lengths[:, 0, 1:-1, 1:-1] - 1,
            spring_lengths[:, 0, 1:-1, 2:] - 1,
            spring_lengths[:, 1, 1:-1, 1:-1] - 1,
            spring_lengths[:, 1, 2:, 1:-1] - 1,
            (spring_lengths[:, 2, 1:-1, 1:-1] - 2 ** (1 / 2))
            * (diagonal_mult) ** (1 / power),
            (spring_lengths[:, 2, 2:, 2:] - 2 ** (1 / 2))
            * (diagonal_mult) ** (1 / power),
            (spring_lengths[:, 3, 1:-1, 1:-1] - 2 ** (1 / 2))
            * (diagonal_mult) ** (1 / power),
            (spring_lengths[:, 3, 2:, 0:-2] - 2 ** (1 / 2))
            * (diagonal_mult) ** (1 / power),
        ]
    )
    # Slightly faster than sum() + pow(), and no need for abs() if power is odd
    result = torch.norm(spring_defs, p=power, dim=0).pow(power)

    total = 4 + 4 * diagonal_mult

    result /= total

    # Remove incorrect smoothness values caused by 2px zero padding
    result[..., 0:2, :] = 0
    result[..., -2:, :] = 0
    result[..., :, 0:2] = 0
    result[..., :, -2:] = 0
    return result
