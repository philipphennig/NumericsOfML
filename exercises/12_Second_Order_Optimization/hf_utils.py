"""This file implements functionality needed for the Hessian-free optimizer. The
functions `L_op`, `R_op`, `hessian_vector_product` and `ggn_vector_product`
are copied from Felix Dangel's `BackPACK`-repo, see
https://github.com/f-dangel/backpack/tree/master/backpack/hessianfree.
"""

import math
from warnings import warn

import torch


def L_op(ys, xs, ws, retain_graph=True, detach=True):
    """Multiplies the vector `ws` with the transposed Jacobian of `ys` w.r.t.
    `xs`."""

    vJ = torch.autograd.grad(
        ys,
        xs,
        grad_outputs=ws,
        create_graph=True,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    if detach:
        return tuple(j.detach() for j in vJ)
    else:
        return vJ


def R_op(ys, xs, vs, retain_graph=True, detach=True):
    """Multiplies the vector `vs` with the Jacobian of `ys` w.r.t. `xs`."""

    if isinstance(ys, tuple):
        ws = [torch.zeros_like(y).requires_grad_(True) for y in ys]
    else:
        ws = torch.zeros_like(ys).requires_grad_(True)

    gs = torch.autograd.grad(
        ys,
        xs,
        grad_outputs=ws,
        create_graph=True,
        retain_graph=retain_graph,
        allow_unused=True,
    )

    re = torch.autograd.grad(
        gs,
        ws,
        grad_outputs=vs,
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )

    if detach:
        return tuple(j.detach() for j in re)
    else:
        return re


def hessian_vector_product(f, params, v, grad_params=None, detach=True):
    """Multiplies the vector `v` with the Hessian, `v = H @ v` where `H` is the
    Hessian of `f` w.r.t. `params`."""
    if grad_params is not None:
        df_dx = tuple(grad_params)
    else:
        df_dx = torch.autograd.grad(
            f, params, create_graph=True, retain_graph=True
        )

    Hv = R_op(df_dx, params, v)

    if detach:
        return tuple(j.detach() for j in Hv)
    else:
        return


def ggn_vector_product(loss, output, plist, v):
    """Multiply a vector with the generalized GGN.

    Args:
        loss: Scalar tensor that represents the loss.
        output: Model output.
        plist: List of trainable parameters whose GGN block is used for
            multiplication.
        v: Vector specified as list of tensors matching the sizes of ``plist``.

    Returns:
        GGN-vector product in list format, i.e. as list that matches the sizes
        of ``plist``.
    """
    Jv = R_op(output, plist, v)
    HJv = hessian_vector_product(loss, output, Jv)
    JTHJv = L_op(output, plist, HJv)
    return JTHJv


def vector_to_parameter_list(vec, parameters):
    """Convert the vector `vec` to a parameter-list format matching
    `parameters`. This function is the inverse of `parameters_to_vector` from
    `torch.nn.utils`. In contrast to `vector_to_parameters`, which replaces the
    value of the parameters, this function leaves the parameters unchanged and
    returns a list of parameter views of the vector.

    Args:
        vec: The vector representing the parameters. This vector is converted to
            a parameter-list format matching `parameters`.
        parameters: An iterable of `torch.Tensor`s containing the parameters.
            These parameters are not changed by the function.

    Raises:
        Warning if not all entries of `vec` are converted.
    """

    if not isinstance(vec, torch.Tensor):
        raise TypeError(f"`vec` should be a torch.Tensor, not {type(vec)}.")

    # Put slices of `vec` into `params_list`
    params_list = []
    pointer = 0
    for param in parameters:
        num_param = param.numel()
        params_list.append(
            vec[pointer : pointer + num_param].view_as(param).data
        )
        pointer += num_param

    # Make sure all entries of the vector have been used (i.e. that `vec` and
    # `parameters` have the same number of elements)
    if pointer != len(vec):
        warn("Not all entries of `vec` have been used.")

    return params_list


def vector_to_trainparams(vec, parameters):
    """Similar to `vector_to_parameters` from `torch.nn.utils`: Replace the
    parameters with the entries of `vec`. But here, the vector `vec` only
    contains the parameter values for the trainable parameters, i.e. those
    parameters with `requires_grad == True`.

    Args:
        vec: The vector representing the trainable parameters.
        parameters: An iterable of `torch.Tensor`s containing the parameters
            (including non-trainable ones).

    Raises:
        Warning, if not all entries of `vec` have been used to fill the
            `parameters`.
    """

    if not isinstance(vec, torch.Tensor):
        raise TypeError(f"`vec` should be a torch.Tensor, not {type(vec)}.")

    # Use slices of `vec` as parameter values but only if they are trainable
    pointer = 0
    for param in parameters:
        num_param = param.numel()
        if param.requires_grad:
            param.data = vec[pointer : pointer + num_param].view_as(param).data
            pointer += num_param

    # Make sure all entries of the vector have been used (i.e. that `vec` and
    # `parameters` have the same number of elements)
    if pointer != len(vec):
        warn("Not all entries of `vec` have been used.")


def cg(A, b, x0=None, maxiter=None, tol=1e-5, atol=1e-8):
    """A PyTorch implementation of the conjugate gradients method.

    Args:
        A: Function that implements matrix-vector products with the positive
            definite system matrix.
        b: A `torch.Tensor` that represents the right-hand side of the linear
            system.
        x0: A `torch.Tensor` that is used as the initial estimate for the
            solution of the linear system. If `None` is given, the zero vector
            is used.
        maxiter: The maximum number of cg iterations. If `None`, the dimension
            of the linear system is used.
        tol, atol: Terminate cg if `norm(residual) <= max(tol * norm(b), atol)`

    Returns:
        An approximate solution to the linear system.

    Raises:
        Warning, if negative directional curvature is detected.
    """

    maxiter = b.numel() if maxiter is None else min(maxiter, b.numel())
    x = torch.zeros_like(b) if x0 is None else x0

    # Initializations
    r = (b - A(x)).detach()
    p = r.clone()
    rs_old = (r**2).sum().item()

    # Stopping criterion
    rs_bound = max([tol * torch.linalg.vector_norm(b).item(), atol])

    # Iterations
    iterations = 0
    while True:
        Ap = A(p).detach()
        pAp = (p * Ap).sum().item()
        if pAp < 0:
            warn(f"Negative curvature detected in iteration {iterations}.")

        alpha = rs_old / pAp
        x.add_(p, alpha=alpha)
        r.sub_(Ap, alpha=alpha)
        rs_new = (r**2).sum().item()
        iterations += 1

        # Check convergence criterion
        if iterations >= maxiter or math.sqrt(rs_new) < rs_bound:
            return x

        p.mul_(rs_new / rs_old)
        p.add_(r)
        rs_old = rs_new
