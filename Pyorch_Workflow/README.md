## What PyTorch Workflow Fundamentals are going to cover:

![image](https://github.com/user-attachments/assets/3d1559a5-025c-43d0-b9cb-1a92330e612c)


| **Topic**                                                    | **Contents**                                                 |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| **1. Getting data ready**                                    | Prepare data & Split data into training and test sets        |
| **2. Building a model**                                      | Subclass **`nn.Module`** to make our model. Notice the model's **parameters** and **forward** computation |
| **3. Fitting the model to data (training)**                  | Set up **loss function**, **optimizer** and use them to build a **training loop**. |
| **4. Making predictions and evaluating a model (inference)** | Using **`inference_mode()`** to make predictions             |
| **5.Not yet**                                                |                                                              |
| **6. Saving and loading a model**                            | You may want to use your model elsewhere. Save the model's **`state_dict()`** to saving the model. |

## 1.Getting data ready

1. Turn your data, whatever it is, into numbers (a representation).
   * Then pick or build a model to learn the representation as best as possible.
2. Split data into training and test sets
   * if you can visualize something, it can do wonders for understanding.

## 2.Building a model

 Create a  model class by Subclass **`nn.Module`**

1. **use PyTorch essential module model building essentials**

​	PyTorch has four (give or take) essential modules you can use to create almost any kind of neural network you can 	imagine.

* | PyTorch module                                               | What does it do?                                             |
  | :----------------------------------------------------------- | :----------------------------------------------------------- |
  | [`torch.nn`](https://pytorch.org/docs/stable/nn.html)        | Contains all of the building blocks for computational graphs (essentially a series of computations executed in a particular way). |
  | [`torch.nn.Parameter`](https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html#parameter) | Stores tensors that can be used with `nn.Module`. If `requires_grad=True` gradients (used for updating model parameters via [**gradient descent**](https://ml-cheatsheet.readthedocs.io/en/latest/gradient_descent.html)) are calculated automatically, this is often referred to as "autograd". |
  | [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) | The base class for all neural network modules, all the building blocks for neural networks are subclasses. If you're building a neural network in PyTorch, your models should subclass `nn.Module`. Requires a `forward()` method be implemented. |
  | [`torch.optim`](https://pytorch.org/docs/stable/optim.html)  | Contains various optimization algorithms (these tell the model parameters stored in `nn.Parameter` how to best change to improve gradient descent and in turn reduce the loss). |
  | `def forward()`                                              | All `nn.Module` subclasses require a `forward()` method, this defines the computation that will take place on the data passed to the particular `nn.Module` (e.g. the linear regression formula above). |

2. **Checking the contents of a PyTorch model**

   * Create a model instance with the class we've made and check its parameters using [`.parameters()`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.parameters).

   * Can also get the state (what the model contains) of the model using [`.state_dict()`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.state_dict).

3. **put our model on the GPU (if it's available).**

```python
# Check model device
next(model_1.parameters()).device
# Set model to GPU if it's available, otherwise it'll default to CPU
model_1.to(device) # the device variable was set above to be "cuda" if available or "cpu" if not
next(model_1.parameters()).device
```

##  3.Training Model

Let the model (try to) find patterns in the (**training**) data.

1. Set up **Loss function & Optimizer** 

   | Function          | What does it do?                                             | Where does it live in PyTorch?                               | Common values                                                |
   | :---------------- | :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
   | **Loss function** | Measures **how wrong your model's predictions** (e.g. `y_preds`) are compared to the truth labels (e.g. `y_test`). Lower the better. | PyTorch has plenty of built-in loss functions in [`torch.nn`](https://pytorch.org/docs/stable/nn.html#loss-functions). | Mean absolute error (MAE) for regression problems ([`torch.nn.L1Loss()`](https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html)). Binary cross entropy for binary classification problems ([`torch.nn.BCELoss()`](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html)). |
   | **Optimizer**     | Tells your model **how to update its internal parameters** to best lower the loss. | You can find various optimization function implementations in [`torch.optim`](https://pytorch.org/docs/stable/optim.html). | Stochastic gradient descent ([`torch.optim.SGD()`](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD)). Adam optimizer ([`torch.optim.Adam()`](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam)). |

2. Create a **Training & Testing loop** for optimization 

   It's train time! 

   do the forward pass, calculate the loss, optimizer zero grad, losssss backwards! Optimizer step step step 

   Let's test now! with torch no grad: do the forward pass, calculate the loss, watch it go down down down!

   | Number | Step name                                   | What does it do?                                             | Code example                      |
   | :----- | :------------------------------------------ | :----------------------------------------------------------- | :-------------------------------- |
   | 1      | Forward pass                                | The model goes through all of the training data once, performing its `forward()` function calculations. | `model(x_train)`                  |
   | 2      | Calculate the loss                          | The model's outputs (predictions) are compared to the ground truth and evaluated to see how wrong they are. | `loss = loss_fn(y_pred, y_train)` |
   | 3      | Zero gradients                              | The optimizers gradients are set to zero (they are accumulated by default) so they can be recalculated for the specific training step. | `optimizer.zero_grad()`           |
   | 4      | Perform backpropagation on the loss         | Computes the gradient of the loss with respect for every model parameter to be updated (each parameter with `requires_grad=True`). This is known as **backpropagation**, hence "backwards". | `loss.backward()`                 |
   | 5      | Update the optimizer (**gradient descent**) | Update the parameters with `requires_grad=True` with respect to the loss gradients in order to improve them. | `optimizer.step()`                |

then testing loop,the testing loop doesn't contain performing backpropagation (`loss.backward()`) or stepping the optimizer (`optimizer.step()`)

| Number | Step name                               | What does it do?                                             | Code example                     |
| :----- | :-------------------------------------- | :----------------------------------------------------------- | :------------------------------- |
| 1      | Forward pass                            | The model goes through all of the testing data once, performing its `forward()` function calculations. | `model(x_test)`                  |
| 2      | Calculate the loss                      | The model's outputs (predictions) are compared to the ground truth and evaluated to see how wrong they are. | `loss = loss_fn(y_pred, y_test)` |
| 3      | Calculate evaluation metrics (optional) | Alongside the loss value you may want to calculate other evaluation metrics such as accuracy on the test set. |                                  |

## 4. Making predictions and evaluating a model (inference)

Just input the X_pred and run the model. There are three things to remember when making predictions (also called performing inference) with a PyTorch model:

1. Set the model in evaluation mode (`model.eval()`).
2. Make the predictions using the inference mode context manager (`with torch.inference_mode(): ...`).
3. All predictions should be made with objects on the same device (e.g. data and model on GPU only or data and model on CPU only).

```python
# 1. Set the model in evaluation mode
model_0.eval()

# 2. Setup the inference mode context manager
with torch.inference_mode():
  # 3. Make sure the calculations are done with the model and data on the same device
  # in our case, we haven't setup device-agnostic code yet so our data and model are
  # on the CPU by default.
  # model_0.to(device)
  # X_test = X_test.to(device)
  y_preds = model_0(X_test)
y_preds
```

## 5. Saving and loading a PyTorch model

For saving and loading models in PyTorch, there are three main methods you should be aware of (all of below have been taken from the [PyTorch saving and loading models guide](https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference)):

| PyTorch method                                               | What does it do?                                             |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| [`torch.save`](https://pytorch.org/docs/stable/torch.html?highlight=save#torch.save) | Saves a serialized object to disk using Python's [`pickle`](https://docs.python.org/3/library/pickle.html) utility. Models, tensors and various other Python objects like dictionaries can be saved using `torch.save`. |
| [`torch.load`](https://pytorch.org/docs/stable/torch.html?highlight=torch load#torch.load) | Uses `pickle`'s unpickling features to deserialize and load pickled Python object files (like models, tensors or dictionaries) into memory. You can also set which device to load the object to (CPU, GPU etc). |
| [`torch.nn.Module.load_state_dict`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=load_state_dict#torch.nn.Module.load_state_dict) | Loads a model's parameter dictionary (`model.state_dict()`) using a saved `state_dict()` object. |

* Saving a PyTorch model's `state_dict()`

  * ```python
    from pathlib import Path
    
    # 1. Create models directory 
    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    
    # 2. Create model save path 
    MODEL_NAME = "01_pytorch_workflow_model_0.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
    
    # 3. Save the model state dict 
    print(f"Saving model to: {MODEL_SAVE_PATH}")
    torch.save(obj=model_0.state_dict(), # only saving the state_dict() only saves the models learned parameters
               f=MODEL_SAVE_PATH) 
    ```

* Loading a saved PyTorch model's `state_dict()`

  * ```python
    # Instantiate a new instance of our model (this will be instantiated with random weights)
    loaded_model_0 = LinearRegressionModel()
    
    # Load the state_dict of our saved model (this will update the new instance of our model with trained weights)
    loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
    ```
