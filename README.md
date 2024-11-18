# NNModuleTools

A collection of neural network utilities.

## Contents

1. npz compare visualizer

    This tool can compare two npz archives visually.

2. float utils

    This tool can inspect the memory of float numbers. It also provides simple arithmetic operations between them.

3. module debugger

    This tool can save all the inputs and outputs of submodules in forward and backward pass. 
    You can also manually save your desired tensors.

## Installation

```shell
python -m pip install nnmoduletools
```

## Usage

1. npz compare visualizer

    See [tutorial.ipynb](tutorial.ipynb)

    Now you can also use command line to compare two npz files and generate a report:

    ```shell
    python -m nnmoduletools.comparer [--tolerance 0.99,0.99] [--abs_tol 1e-8] [--rel-tol 1e-3] [--verbose 3] [--output_dir compare_report] [--output_fn compare_report.md] [--info] [--dump [10]] target.npz ref.npz [tensor1 tensor2 ...] 
    ```

    See `python -m nnmoduletools.comparer --help` for more information.

    The command line tool only support basic operations.  You can also use `Reporter` in Python scripts to generate your own report:

    ```Python
    from nnmoduletools.comparer import Reporter
    with Reporter("path/to/output/dir", "report.md"):
        comparer = nnmoduletools.NPZComparer("path/to/target.npz", "path/to/ref.npz")
        print("# Compare Report: tensor")
        comparer.plot_vs_auto(tensor="tensor", save_fig=True, save_dir="subdir")
        comparer.dump_vs_plot(top_k=20)
    ```

    You will have your report in `path/to/output/dir/report.md` and the plots in `path/to/output/dir/subdir/`.

2. float utils

    See [tutorial.ipynb](tutorial.ipynb)

3. module debugger

    Typical usage: run the same module using two different devices, dump the tensors into npz file and compare them in npz compare visualizer.

    You may have to set environment variables:
    ```shell
    export DBG_DEVICE=cuda # suppose you are using cuda as the device; if you are using deepspeed, device name will be automatically get from deepspeed.get_accelerator()
    export DBG_SAVE_ALL=1 # by default no tensors are saved; you should export this to enable saving
    ```

    Insert these lines into your code:
    ```Python
    import torch
    from nnmoduletools.module_debugger import register_hook, save_tensors, save_model_params, save_model_grads, combine_npz

    class YourModule(torch.nn.Module):
        ...
        # Your module here

    model = YourModule()
    model.apply(register_hook) # <=== apply the hook to print log and save input output tensors
    ...
    save_model_params(model, 0) # <=== save the model params before training
    # in your training loop
    for step in range(1, total_steps+1):
        output = model.forward(input)
        loss = loss_function(output, target)
        loss.backward()
        combine_npz(step) # <=== combine input and output tensors into large npzs
        save_model_grads(model, step) # <=== save the model grads after backward pass
        optimizer.step()
        save_model_params(model, step) # <=== save the model params after optim update
        save_tensors(tensor_to_save, name_to_save, dir_to_save, save_grad_instead) # <=== save the tensor you want to given directory. You can save grad instead by passing save_grad_instead=True

    ```

    The log and npz files will be saved in a directory named like `logs_2024.06.04.16.52.12/`

    You can get the latest log directory easily with `LogReader`:
    ```Python
    from nnmoduletools.module_debugger import LogReader
    latest = LogReader(devices=["cuda"])
    print(latest.cuda_dir)
    ```

## Troubleshooting

You may encounter such errors when using backward hooks:
```shell
RuntimeError: Output 0 of BackwardHookFunctionBackward is a view and is being modified inplace. This view was created inside a custom Function (or because an input was returned as-is) and the autograd logic to handle view+inplace would override the custom backward associated with the custom Function, leading to incorrect gradients. This behavior is forbidden. You can fix this by cloning the output of the custom Function.
```

It is likely because that your module has an inplace operation right before return in forward, such as 

```Python
class YourModule(torch.nn.Module):
    ...
    def forward(self, x):
        result = ...
        result += 1 # <===
        return result
```

You can replace the operation with

```Python
    def forward(self, x):
        result = ...
        result = result + 1 # <===
        return result
```
    

    
