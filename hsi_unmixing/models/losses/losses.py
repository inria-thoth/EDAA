import torch


def ASC_penalty(alpha, nu):
    return (nu / 2) * ((alpha.sum(1) - 1) ** 2).mean(0)
