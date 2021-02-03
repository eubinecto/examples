## experiment 1
- teacher: hidden units=800
- student: hidden units=400 (half of the teacher)

```markdown
/Users/eubin/Desktop/Study/examples/examplesenv/bin/python /Users/eubin/Desktop/Study/examples/distill/train_kd.py
Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to mnist_data/MNIST/raw/train-images-idx3-ubyte.gz
 94%|█████████▎| 9273344/9912422 [00:03<00:00, 3673475.54it/s]Extracting mnist_data/MNIST/raw/train-images-idx3-ubyte.gz to mnist_data/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to mnist_data/MNIST/raw/train-labels-idx1-ubyte.gz

0it [00:00, ?it/s]
  0%|          | 0/28881 [00:00<?, ?it/s]

0it [00:00, ?it/s]Extracting mnist_data/MNIST/raw/train-labels-idx1-ubyte.gz to mnist_data/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to mnist_data/MNIST/raw/t10k-images-idx3-ubyte.gz


  0%|          | 0/1648877 [00:00<?, ?it/s]

  1%|          | 16384/1648877 [00:01<00:53, 30477.72it/s]

  9%|▉         | 147456/1648877 [00:01<00:05, 296133.85it/s]

 19%|█▉        | 311296/1648877 [00:01<00:02, 601206.29it/s]

 26%|██▋       | 434176/1648877 [00:01<00:01, 722549.42it/s]

 45%|████▌     | 745472/1648877 [00:01<00:00, 1319004.74it/s]

 63%|██████▎   | 1032192/1648877 [00:01<00:00, 1725532.54it/s]

 92%|█████████▏| 1523712/1648877 [00:01<00:00, 2597399.13it/s]Extracting mnist_data/MNIST/raw/t10k-images-idx3-ubyte.gz to mnist_data/MNIST/raw



0it [00:00, ?it/s]Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to mnist_data/MNIST/raw/t10k-labels-idx1-ubyte.gz



  0%|          | 0/4542 [00:00<?, ?it/s]/Users/eubin/Desktop/Study/examples/examplesenv/lib/python3.8/site-packages/torchvision/datasets/mnist.py:480: UserWarning: The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors. This means you can write to the underlying (supposedly non-writeable) NumPy array using the tensor. You may want to copy the array to protect its data or make it writeable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at  ../torch/csrc/utils/tensor_numpy.cpp:141.)
Extracting mnist_data/MNIST/raw/t10k-labels-idx1-ubyte.gz to mnist_data/MNIST/raw
Processing...
  return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)
Done!
32768it [00:03, 9494.56it/s]             
1654784it [00:02, 609582.22it/s]                              
8192it [00:00, 8536.08it/s]             
Training Teacher... 
9920512it [00:20, 3673475.54it/s]                             Epoch: 1, Loss: 923.1929931640625, Accuracy: 0.8744333333333333
Epoch: 2, Loss: 408.743896484375, Accuracy: 0.9375666666666667
Epoch: 3, Loss: 302.6789855957031, Accuracy: 0.9534833333333333
Epoch: 4, Loss: 238.3784637451172, Accuracy: 0.96375
Epoch: 5, Loss: 194.31005859375, Accuracy: 0.9704333333333334
Training Student...
Epoch: 1, Loss: 764.597748234868, Accuracy: 0.82525
Epoch: 2, Loss: 296.5063241943717, Accuracy: 0.9159
Epoch: 3, Loss: 232.86592195741832, Accuracy: 0.9321
Epoch: 4, Loss: 190.29843407310545, Accuracy: 0.9437166666666666
Epoch: 5, Loss: 159.67518485244364, Accuracy: 0.9525333333333333
--------------------------------------------------------------------------------
Accuracy: **0.9535**
--------------------------------------------------------------------------------
Total parameters for the teacher network are: 1276810
Total parameters for the student network are: 478410
9920512it [03:55, 42078.92it/s]  

Process finished with exit code 0

```


## experiment 2
- teacher: 800
- student: 200 (a quarter of the teacher)

```markdown
/Users/eubin/Desktop/Study/examples/examplesenv/bin/python /Users/eubin/Desktop/Study/examples/distill/train_kd.py
Training Teacher... 
Epoch: 1, Loss: 909.0924072265625, Accuracy: 0.87815
Epoch: 2, Loss: 404.0888977050781, Accuracy: 0.93765
Epoch: 3, Loss: 295.6884460449219, Accuracy: 0.9544333333333334
Epoch: 4, Loss: 230.5728302001953, Accuracy: 0.9643666666666667
Epoch: 5, Loss: 187.2537078857422, Accuracy: 0.9710833333333333
Training Student...
Epoch: 1, Loss: 792.8302914723754, Accuracy: 0.8212166666666667
Epoch: 2, Loss: 305.89222536608577, Accuracy: 0.9128333333333334
Epoch: 3, Loss: 242.50045359693468, Accuracy: 0.9301
Epoch: 4, Loss: 200.73941019177437, Accuracy: 0.94065
Epoch: 5, Loss: 170.83086107578129, Accuracy: 0.9485
--------------------------------------------------------------------------------
Accuracy: **0.9499**
--------------------------------------------------------------------------------
Total parameters for the teacher network are: 1276810
Total parameters for the student network are: 199210

Process finished with exit code 0
```