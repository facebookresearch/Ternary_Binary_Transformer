# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import math

class ElasticQuantBinarizerSigned(torch.autograd.Function):
    """
        Modified from Learned Step-size Quantization.
        https://arxiv.org/abs/1902.08153
    """
    @staticmethod
    def forward(ctx, input, alpha, num_bits, layerwise):
        """
        :param input: input to be quantized
        :param alpha: the step size
        :param num_bits: quantization bits
        :param layerwise: rowwise quant
        :return: quantized output
        """
        if not layerwise:
            # TODO
            raise NotImplementedError
        ctx.num_bits = num_bits
        if num_bits == 32:
            return input
        if num_bits == 1 or num_bits == 2:
            Qn = -1
            Qp = 1
        else:
            Qn = -2 ** (num_bits - 1)
            Qp = 2 ** (num_bits - 1) - 1

        eps = torch.tensor(0.00001).float().to(alpha.device)
        if alpha.item() == 1.0 and (not alpha.initialized):
            alpha.initialize_wrapper(input, num_bits, symmetric=True, init_method='default')
        alpha = torch.where(alpha > eps, alpha, eps)
        assert alpha > 0, 'alpha = {:.6f} becomes non-positive'.format(alpha)

        grad_scale = 1.0 / math.sqrt(input.numel()) if not Qp else 1.0 / math.sqrt(input.numel() * Qp)
        ctx.save_for_backward(input, alpha)
        ctx.other = grad_scale, Qn, Qp
        if num_bits == 1:
            q_w = input.sign()
        else:
            q_w = (input / alpha).round().clamp(Qn, Qp)
        w_q = q_w * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits == 32:
            return grad_output, None, None, None

        input_, alpha = ctx.saved_tensors
        grad_scale, Qn, Qp = ctx.other
        q_w = input_ / alpha
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = 1.0 - indicate_small - indicate_big # this is more cpu-friendly than torch.ones(input_.shape)
        if ctx.num_bits == 1:
            grad_alpha = ((input_.sign()) * grad_output * grad_scale).sum().unsqueeze(dim=0)
        else:
            grad_alpha = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (
                    -q_w + q_w.round())) * grad_output * grad_scale).sum().unsqueeze(dim=0)
        grad_input = indicate_middle * grad_output
        return grad_input, grad_alpha, None, None


class ElasticQuantBinarizerUnsigned(torch.autograd.Function):
    """
        Modified from Learned Step-size Quantization.
        https://arxiv.org/abs/1902.08153
    """
    @staticmethod
    def forward(ctx, input, alpha, num_bits, layerwise):
        """
        :param input: input to be quantized
        :param alpha: the step size
        :param num_bits: quantization bits
        :param layerwise: rowwise quant
        :return: quantized output
        """
        if not layerwise:
            # TODO
            raise NotImplementedError
        ctx.num_bits = num_bits
        if num_bits == 32:
            return input
        Qn = 0
        if num_bits == 2:
            Qp = 2
        else:
            Qp = 2 ** (num_bits) - 1

        if num_bits == 1:
            input_ = input
        else:
            min_val = input.min().item()
            input_ = input - min_val

        eps = torch.tensor(0.00001).float().to(alpha.device)
        if alpha.item() == 1.0 and (not alpha.initialized):
            alpha.initialize_wrapper(input, num_bits, symmetric=False, init_method='default')
        alpha = torch.where(alpha > eps, alpha, eps)
        assert alpha > 0, 'alpha = {:.6f} becomes non-positive'.format(alpha)

        grad_scale = 1.0 / math.sqrt(input.numel() * Qp)
        ctx.save_for_backward(input_, alpha)
        ctx.other = grad_scale, Qn, Qp
        q_w = (input_ / alpha).round().clamp(Qn, Qp)
        w_q = q_w * alpha
        if num_bits != 1:
            w_q = w_q + min_val

        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits == 32:
            return grad_output, None, None, None

        input_, alpha = ctx.saved_tensors
        grad_scale, Qn, Qp = ctx.other
        q_w = input_ / alpha
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = 1.0 - indicate_small - indicate_big   # this is more cpu-friendly than torch.ones(input_.shape)
        grad_alpha = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (
                -q_w + q_w.round())) * grad_output * grad_scale).sum().unsqueeze(dim=0)
        grad_input = indicate_middle * grad_output
        return grad_input, grad_alpha, None, None

class AlphaInit(nn.Parameter):
    def __init__(self, tensor):
        super(AlphaInit, self).__new__(nn.Parameter, data=tensor)
        self.initialized = False

    def _initialize(self, init_tensor):
        assert not self.initialized, 'already initialized.'
        self.data.copy_(init_tensor)
        self.initialized = True

    def initialize_wrapper(self, tensor, num_bits, symmetric, init_method='default'):
        Qp = 2 ** (num_bits - 1) - 1 if symmetric else 2 ** (num_bits) - 1
        if Qp == 0:
            Qp = 1.0
        if init_method == 'default':
            init_val = 2 * tensor.abs().mean() / math.sqrt(Qp) if symmetric \
                else 4 * tensor.abs().mean() / math.sqrt(Qp)
        elif init_method == 'uniform':
            init_val = 1./(2*Qp+1) if symmetric else 1./Qp

        self._initialize(init_val)

class SymQuantizer(torch.autograd.Function):
    """
        uniform quantization
    """
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise):
        """
        :param ctx:
        :param input: tensor to be quantized
        :param clip_val: clip the tensor before quantization
        :param quant_bits: number of bits
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)
        input = torch.clamp(input, clip_val[0], clip_val[1])
        #input = torch.where(input < clip_val[1], input, clip_val[1])
        #input = torch.where(input > clip_val[0], input, clip_val[0])
        # NOTE: dynamic scaling (max_input).
        if layerwise:
            max_input = torch.max(torch.abs(input)).expand_as(input)
        else:
            if input.ndimension() <= 3:
                # weight & hidden layer
                max_input = torch.max(torch.abs(input), dim=-1, keepdim=True)[0].expand_as(input).detach()
            elif input.ndimension() == 4:
                # TODO: attention score matrix, calculate alpha / beta per head
                tmp = input.view(input.shape[0], input.shape[1], -1)
                max_input = torch.max(torch.abs(tmp), dim=-1, keepdim=True)[0].unsqueeze(-1).expand_as(input).detach()
            else:
                raise ValueError
        s = (2 ** (num_bits - 1) - 1) / max_input
        output = torch.round(input * s).div(s)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None


class AsymQuantizer(torch.autograd.Function):
    """
        min-max quantization
    """
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise):
        """
        :param ctx:
        :param input: tensor to be quantized
        :param clip_val: clip the tensor before quantization
        :param quant_bits: number of bits
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)

        input = torch.where(input < clip_val[1], input, clip_val[1])
        input = torch.where(input > clip_val[0], input, clip_val[0])
        # input = torch.clamp(input, clip_val[0], clip_val[1])
        # NOTE: dynamic scaling gives better performance than static
        if layerwise:
            alpha = (input.max() - input.min()).detach()
            beta = input.min().detach()
        else:
            if input.ndimension() <= 3:
                # weight & hidden layer
                alpha = (input.max(dim=-1, keepdim=True)[0] - input.min(dim=-1, keepdim=True)[0]).expand_as(input).detach()
                beta = input.min(dim=-1, keepdim=True)[0].expand_as(input).detach()
            elif input.ndimension() == 4:
                # TODO: attention score matrix, calculate alpha / beta per head
                tmp = input.view(input.shape[0], input.shape[1], -1)
                alpha = (tmp.max(dim=-1, keepdim=True)[0].unsqueeze(-1) - \
                            tmp.min(dim=-1, keepdim=True)[0].unsqueeze(-1)).expand_as(input).detach()
                beta = tmp.min(dim=-1, keepdim=True)[0].unsqueeze(-1).expand_as(input).detach()
            else:
                raise ValueError
        input_normalized = (input - beta) / (alpha + 1e-8)
        s = (2**num_bits - 1)
        quant_input = torch.round(input_normalized * s).div(s)
        output = quant_input * (alpha + 1e-8) + beta


        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None


class TwnQuantizer(torch.autograd.Function):
    """Ternary Weight Networks (TWN)
    Ref: https://arxiv.org/abs/1605.04711
    """

    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise):
        """
        :param input: tensor to be ternarized
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)
        input = torch.where(input < clip_val[1], input, clip_val[1])
        input = torch.where(input > clip_val[0], input, clip_val[0])
        if layerwise:
            m = input.norm(p=1).div(input.nelement())
            thres = 0.7 * m
            pos = (input > thres).float()
            neg = (input < -thres).float()
            mask = (input.abs() > thres).float()
            alpha = (mask * input).abs().sum() / mask.sum()
            result = alpha * pos - alpha * neg
        else: # row-wise only for embed / weight
            n = input[0].nelement()
            m = input.data.norm(p=1, dim=1).div(n)
            thres = (0.7 * m).view(-1, 1).expand_as(input)
            pos = (input > thres).float()
            neg = (input < -thres).float()
            mask = (input.abs() > thres).float()
            alpha = ((mask * input).abs().sum(dim=1) / mask.sum(dim=1)).view(-1, 1)
            result = alpha * pos - alpha * neg

        return result

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None

class BwnQuantizer(torch.autograd.Function):
    """Binary Weight Network (BWN)
     Ref: https://arxiv.org/abs/1603.05279
     """

    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise):
        """
        :param input: tensor to be binarized
        :return: quantized tensor
        """
        ctx.save_for_backward(input)
        if layerwise:
            s = input.size()
            m = input.norm(p=1).div(input.nelement())
            e = input.mean()
            result = (input-e).sign().mul(m.expand(s))
        else:
            n = input[0].nelement()  # W of size axb, return a vector of  ax1
            s = input.size()
            m = input.norm(1, 1, keepdim=True).div(n)
            e = input.mean()
            result = (input-e).sign().mul(m.expand(s))

        return result

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        grad_input = grad_output.clone()
        return grad_input, None, None, None

class QuantizeLinear(nn.Linear):

    def __init__(self,  *kargs, symmetric=True, bias=True, config=None):
        super(QuantizeLinear, self).__init__(*kargs,bias=True)
        self.weight_bits = config.weight_bits
        self.quantize_act = config.quantize_act
        #params for weight quant
        self.register_buffer('weight_clip_val', torch.tensor([config.clip_val]))
        if self.quantize_act:
            self.input_bits = config.input_bits
            if self.input_bits <= 2 and symmetric:
                self.act_clip_val = AlphaInit(torch.tensor(1.0))
                self.act_quantizer = ElasticQuantBinarizerSigned
            elif self.input_bits <= 2 and not symmetric:
                self.act_clip_val = AlphaInit(torch.tensor(1.0))
                self.act_quantizer = ElasticQuantBinarizerUnsigned
            elif self.input_bits == 8 and symmetric:
                self.register_buffer('act_clip_val', torch.tensor([-config.clip_val, config.clip_val]))
                self.act_quantizer = SymQuantizer
            elif self.input_bits == 8 and not symmetric:
                self.register_buffer('act_clip_val', torch.tensor([-config.clip_val, config.clip_val]))
                self.act_quantizer = AsymQuantizer
            else:
                raise NotImplementedError

    def forward(self, input):
        # quantize weight
        assert len(self.weight.size()) == 2
        real_weights = self.weight
        if self.weight_bits == 1:
            scaling_factor = torch.mean(abs(real_weights), dim=1, keepdim=True).detach()
            quan_weights_no_grad = scaling_factor * (torch.sign(real_weights/scaling_factor))
        elif self.weight_bits == 2:
            scaling_factor = 4/3 * torch.mean(abs(real_weights), dim=1, keepdim=True).detach()
            quan_weights_no_grad = scaling_factor * (torch.round(torch.clamp(real_weights/scaling_factor, -1, 1)))
        else:
            raise NotImplementedError

        weight = quan_weights_no_grad.detach() - real_weights.detach() + real_weights
        # quantize input
        input = self.act_quantizer.apply(input, self.act_clip_val, self.input_bits, True)

        out = nn.functional.linear(input, weight)
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)

        return out


class QuantizeEmbedding(nn.Embedding):

    def __init__(self,  *kargs,padding_idx=None, config = None):
        print('init quantize emb')
        super(QuantizeEmbedding, self).__init__(*kargs, padding_idx = padding_idx)
        self.weight_bits = config.weight_bits
        self.layerwise = False
        self.register_buffer('weight_clip_val', torch.tensor([-config.clip_val, config.clip_val]))

    def forward(self, input):
        assert len(self.weight.size()) == 2
        real_weights = self.weight
        if self.weight_bits == 1:
            scaling_factor = torch.mean(abs(real_weights), dim=1, keepdim=True).detach()
            quan_weights_no_grad = scaling_factor * (torch.sign(real_weights/scaling_factor))
        elif self.weight_bits == 2:
            scaling_factor = 4/3 * torch.mean(abs(real_weights), dim=1, keepdim=True).detach()
            quan_weights_no_grad = scaling_factor * (torch.round(torch.clamp(real_weights/scaling_factor, -1, 1)))
        else:
            raise NotImplementedError

        weight = quan_weights_no_grad.detach() - real_weights.detach() + real_weights

        out = nn.functional.embedding(
            input, weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)
        return out
