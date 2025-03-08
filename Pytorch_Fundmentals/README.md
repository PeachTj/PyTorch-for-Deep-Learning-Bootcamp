## 1.Creating tensors

**`size & dimension`**

[**`DataType`**](https://pytorch.org/docs/stable/tensors.html#data-types) 

* scalar, vector, matrix 
* zeros, ones
* Creating tensors using `torch.arange`,`zeros_like`
*  create some tensors with specific datatypes

## 2.Getting information from tensors

- `shape` - what shape is the tensor? (some operations require specific shape rules)
- `dtype` - what datatype are the elements within the tensor stored in?
- `device` - what device is the tensor stored on? (usually GPU or CPU)

## 3.Manipulating tensors

- Addition,Substraction
- Multiplication (element-wise)
- Matrix multiplication
- aggregation

## 4.Dealing with tensor shapes

These methods help you make sure the right elements of your tensors are mixing with the right elements of other tensors.

| Method                                                       | One-line description                                         |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| [`torch.reshape(input, shape)`](https://pytorch.org/docs/stable/generated/torch.reshape.html#torch.reshape) | Reshapes `input` to `shape` (if compatible), can also use `torch.Tensor.reshape()`. |
| [`Tensor.view(shape)`](https://pytorch.org/docs/stable/generated/torch.Tensor.view.html) | Returns a view of the original tensor in a different `shape` but shares the same data as the original tensor. |
| [`torch.stack(tensors, dim=0)`](https://pytorch.org/docs/1.9.1/generated/torch.stack.html) | Concatenates a sequence of `tensors` along a new dimension (`dim`), all `tensors` must be same size. |
| [`torch.squeeze(input)`](https://pytorch.org/docs/stable/generated/torch.squeeze.html) | Squeezes `input` to remove all the dimenions with value `1`. |
| [`torch.unsqueeze(input, dim)`](https://pytorch.org/docs/1.9.1/generated/torch.unsqueeze.html) | Returns `input` with a dimension value of `1` added at `dim`. |
| [`torch.permute(input, dims)`](https://pytorch.org/docs/stable/generated/torch.permute.html) | Returns a *view* of the original `input` with its dimensions permuted (rearranged) to `dims`. |

## 5.Indexing Tensors

Select specific data from tensors 

* `[i],[:]`

## 6.Match Pytorch Tensors and Numpy

- [`torch.from_numpy(ndarray)`](https://pytorch.org/docs/stable/generated/torch.from_numpy.html) - NumPy array -> PyTorch tensor.
- [`torch.Tensor.numpy()`](https://pytorch.org/docs/stable/generated/torch.Tensor.numpy.html) - PyTorch tensor -> NumPy array.

## 7.Reproducibility & Random Seed

use Random seed to Perform repeatable experiments.

That's where [`torch.manual_seed(seed)`](https://pytorch.org/docs/stable/generated/torch.manual_seed.html) comes in,




