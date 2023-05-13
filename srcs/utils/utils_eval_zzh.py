
import torch
import numpy as np
import os
import sys
from tqdm import tqdm
from ptflops import get_model_complexity_info


# ===============
# Model Information
# ===============


# ------------------------------------------
# parameter number
# ------------------------------------------
def param_counting(model, logger=None):
    """
    get the number of parameters in the model

    Args:
        model (nn.Module): model
        logger (optional): logging logger. Defaults to None.
    """
    total_num = sum(np.prod(p.size()) for p in model.parameters())
    trainable_num = sum(np.prod(p.size())
                        for p in model.parameters() if p.requires_grad)

    if logger:
        logger.info(
            '='*40+f'\nParam. Num: total {total_num:.2f} M, trainable {trainable_num:.2f} M\n'+'='*40)
    else:
        print((
            '='*40+f'\nTrainable parameters: {trainable_num:.2f} M\n'+'='*40))

# --------------------------------------------
# print MACs & Param. Num
# --------------------------------------------


def model_complexity(model, input_shape, input_constructor=None, print_per_layer_stat=False, logger=None, **kwargs):
    """
    calcalate the MACs and parameters of the model based on ptflops package

    Args:
        model (nn.Module): model
        input_shape (tuple | list): the shape of the input sample (batch size not included)
        input_constructor (optional): input batch constructor. Defaults to None.
            If None, the input batch will be generated by `torch.ones(()).new_empty(...)` with the given `input_shape`
            If not None, the input batch will be generated by `input_constructor(input_shape)`. Note that the customized `input_constructor` function should takes `input_shape` as the input and return a dict of kwargs for the model's forward function. For example,
            ```
            def input_constructor(in_shape):
                input_kwargs = {'x': torch.randn((1, *in_shape), device=model.device)}
                return input_kwargs
            ```
        logger (optional): logging logger. Defaults to None.
        kwargs: other arguments for ptflops.get_model_complexity_info
    """

    macs, params = get_model_complexity_info(
        model=model, input_res=input_shape, input_constructor=input_constructor, print_per_layer_stat=print_per_layer_stat, **kwargs)  #
    if logger:
        logger.info(
            '='*40+'\n{:<30} {}'.format('Inputs resolution: ', input_shape))
        logger.info(
            '{:<30} {}'.format('Computational complexity: ', macs))
        logger.info('{:<30}  {}\n'.format(
            'Number of parameters: ', params)+'='*40)
    else:
        print('='*40+'\n{:<30} {}'.format('Inputs resolution: ', input_shape))
        print('{:<30} {}'.format('Computational complexity: ', macs))
        print('{:<30}  {}\n'.format(
            'Number of parameters: ', params)+'='*40)

# ---------------------------------------------------
# inference time
# ---------------------------------------------------

def gpu_inference_time(model, input_shape, logger=None, device=None, repetitions=100):
    """
    inference time estimation

    Args:
        model: torch model
        input_shape (list | tuple): shape of the model's batch inputs
        logger: logger. Defaults to None
        device: GPU cuda device. Defaults to None, i.e. use model's woring device
        repetitions (int, optional): testing times. Defaults to 100.
    """

    # INIT
    if device is None:
        if next(model.parameters()).is_cuda:
            device = next(model.parameters()).device
        else:
            raise ValueError("Please assign a GPU device for inference")
    else:
        model.to(device)

    if isinstance(input_shape, list):
        dummy_input = [torch.randn(shape_k, dtype=torch.float).to(
            device) for shape_k in input_shape]
    elif isinstance(input_shape, tuple):
        dummy_input = [torch.randn(input_shape, dtype=torch.float).to(
            device)]
    else:
        raise ValueError(
            f"`input_shape` should be a tuple or a list containing multiple tuples, but get `{input_shape}` ")

    starter, ender = torch.cuda.Event(
        enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((repetitions, 1))

    # GPU-WARM-UP
    for _ in range(10):
        _ = model(*dummy_input)

    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in tqdm(range(repetitions), desc='Inference Time Est:'):
            starter.record()
            _ = model(*dummy_input)
            ender.record()

            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    mean_time = np.sum(timings) / repetitions
    std_time = np.std(timings)
    if logger:
        logger.info(
            '='*40+f'\nInference Time Estimation \nInputs Shape:\t{input_shape} \nEstimated Time:\t{mean_time:.3f}ms \nEstimated Std:\t{std_time:.3f}ms\n'+'='*40)
    else:
        print(
            '='*40+f'\nInference Time Estimation \nInputs Shape:\t{input_shape} \nEstimated Time:\t{mean_time:.3f}ms \nEstimated Std:\t{std_time:.3f}ms\n'+'='*40)


if __name__ == '__main__':

    model = DeepRFT()
    device = torch.device("cuda:0")
    input_shape = (1, 3, 224, 224)
    inference_time_est(model, device, input_shape)
