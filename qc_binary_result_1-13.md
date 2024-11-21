(quantum) jamie@Dell:~/Works/Waterloo/ECE730/ECE730Project$ ^C
(quantum) jamie@Dell:~/Works/Waterloo/ECE730/ECE730Project$ python qc_binary.py 
2024-11-20 19:50:04.758743: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:936] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2024-11-20 19:50:04.764685: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:936] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2024-11-20 19:50:04.765047: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:936] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2024-11-20 19:50:04.765569: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2024-11-20 19:50:04.767802: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:936] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2024-11-20 19:50:04.768198: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:936] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2024-11-20 19:50:04.768501: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:936] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2024-11-20 19:50:05.184923: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:936] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2024-11-20 19:50:05.185285: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:936] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2024-11-20 19:50:05.185596: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:936] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2024-11-20 19:50:05.185886: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1525] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 4126 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 3060 Laptop GPU, pci bus id: 0000:01:00.0, compute capability: 8.6
Loading data from pickle files...
Processing training data...
Processing test data...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4380/4380 [00:00<00:00, 18182.52it/s]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1440/1440 [00:00<00:00, 7187.95it/s]
Number of original training examples: 4380
Number of original test examples: 1440
2024-11-20 19:50:05.797166: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 27471360 exceeds 10% of free system memory.
Epoch 1/40
137/137 [==============================] - 35s 251ms/step - loss: 0.9915 - hinge_accuracy: 0.5197 - val_loss: 0.9832 - val_hinge_accuracy: 0.5208
Epoch 2/40
137/137 [==============================] - 34s 251ms/step - loss: 0.9817 - hinge_accuracy: 0.5316 - val_loss: 0.9688 - val_hinge_accuracy: 0.5285
Epoch 3/40
137/137 [==============================] - 35s 252ms/step - loss: 0.9593 - hinge_accuracy: 0.5392 - val_loss: 0.9470 - val_hinge_accuracy: 0.5375
Epoch 4/40
137/137 [==============================] - 35s 254ms/step - loss: 0.9297 - hinge_accuracy: 0.5581 - val_loss: 0.9329 - val_hinge_accuracy: 0.5347
Epoch 5/40
137/137 [==============================] - 35s 254ms/step - loss: 0.9079 - hinge_accuracy: 0.5603 - val_loss: 0.9272 - val_hinge_accuracy: 0.5472
Epoch 6/40
137/137 [==============================] - 35s 253ms/step - loss: 0.8971 - hinge_accuracy: 0.5575 - val_loss: 0.9259 - val_hinge_accuracy: 0.5257
Epoch 7/40
137/137 [==============================] - 35s 254ms/step - loss: 0.8922 - hinge_accuracy: 0.5534 - val_loss: 0.9258 - val_hinge_accuracy: 0.5236
Epoch 8/40
137/137 [==============================] - 35s 253ms/step - loss: 0.8890 - hinge_accuracy: 0.5553 - val_loss: 0.9261 - val_hinge_accuracy: 0.5375
Epoch 9/40
137/137 [==============================] - 35s 252ms/step - loss: 0.8871 - hinge_accuracy: 0.5573 - val_loss: 0.9276 - val_hinge_accuracy: 0.5222
Epoch 10/40
137/137 [==============================] - 35s 253ms/step - loss: 0.8858 - hinge_accuracy: 0.5546 - val_loss: 0.9267 - val_hinge_accuracy: 0.5347
Epoch 11/40
137/137 [==============================] - 34s 251ms/step - loss: 0.8835 - hinge_accuracy: 0.5578 - val_loss: 0.9259 - val_hinge_accuracy: 0.5465
Epoch 12/40
137/137 [==============================] - 35s 253ms/step - loss: 0.8827 - hinge_accuracy: 0.5563 - val_loss: 0.9270 - val_hinge_accuracy: 0.5410
Epoch 13/40
137/137 [==============================] - 35s 253ms/step - loss: 0.8822 - hinge_accuracy: 0.5578 - val_loss: 0.9276 - val_hinge_accuracy: 0.5312
Epoch 14/40
137/137 [==============================] - 34s 251ms/step - loss: 0.8807 - hinge_accuracy: 0.5623 - val_loss: 0.9264 - val_hinge_accuracy: 0.5375
Epoch 15/40
137/137 [==============================] - 34s 250ms/step - loss: 0.8788 - hinge_accuracy: 0.5665 - val_loss: 0.9263 - val_hinge_accuracy: 0.5424
Epoch 16/40
137/137 [==============================] - 34s 250ms/step - loss: 0.8765 - hinge_accuracy: 0.5642 - val_loss: 0.9263 - val_hinge_accuracy: 0.5278
Epoch 17/40
137/137 [==============================] - 34s 251ms/step - loss: 0.8748 - hinge_accuracy: 0.5668 - val_loss: 0.9251 - val_hinge_accuracy: 0.5257
Epoch 18/40
137/137 [==============================] - 34s 251ms/step - loss: 0.8717 - hinge_accuracy: 0.5659 - val_loss: 0.9227 - val_hinge_accuracy: 0.5389
Epoch 19/40
137/137 [==============================] - 34s 251ms/step - loss: 0.8695 - hinge_accuracy: 0.5717 - val_loss: 0.9207 - val_hinge_accuracy: 0.5347
Epoch 20/40
137/137 [==============================] - 34s 251ms/step - loss: 0.8663 - hinge_accuracy: 0.5746 - val_loss: 0.9160 - val_hinge_accuracy: 0.5347
Epoch 21/40
137/137 [==============================] - 35s 253ms/step - loss: 0.8647 - hinge_accuracy: 0.5737 - val_loss: 0.9142 - val_hinge_accuracy: 0.5354
Epoch 22/40
137/137 [==============================] - 34s 251ms/step - loss: 0.8602 - hinge_accuracy: 0.5738 - val_loss: 0.9101 - val_hinge_accuracy: 0.5410
Epoch 23/40
137/137 [==============================] - 34s 251ms/step - loss: 0.8579 - hinge_accuracy: 0.5746 - val_loss: 0.9077 - val_hinge_accuracy: 0.5361
Epoch 24/40
137/137 [==============================] - 34s 250ms/step - loss: 0.8545 - hinge_accuracy: 0.5761 - val_loss: 0.9021 - val_hinge_accuracy: 0.5514
Epoch 25/40
137/137 [==============================] - 34s 250ms/step - loss: 0.8517 - hinge_accuracy: 0.5842 - val_loss: 0.9018 - val_hinge_accuracy: 0.5306
Epoch 26/40
137/137 [==============================] - 34s 251ms/step - loss: 0.8488 - hinge_accuracy: 0.5833 - val_loss: 0.8952 - val_hinge_accuracy: 0.5396
Epoch 27/40
137/137 [==============================] - 34s 251ms/step - loss: 0.8468 - hinge_accuracy: 0.5883 - val_loss: 0.8878 - val_hinge_accuracy: 0.5451
Epoch 28/40
137/137 [==============================] - 35s 257ms/step - loss: 0.8445 - hinge_accuracy: 0.5883 - val_loss: 0.8819 - val_hinge_accuracy: 0.5604
Epoch 29/40
137/137 [==============================] - 35s 258ms/step - loss: 0.8422 - hinge_accuracy: 0.5902 - val_loss: 0.8890 - val_hinge_accuracy: 0.5528
Epoch 30/40
137/137 [==============================] - 36s 260ms/step - loss: 0.8397 - hinge_accuracy: 0.5906 - val_loss: 0.8704 - val_hinge_accuracy: 0.5708
Epoch 31/40
137/137 [==============================] - 35s 254ms/step - loss: 0.8365 - hinge_accuracy: 0.5974 - val_loss: 0.8702 - val_hinge_accuracy: 0.5639
Epoch 32/40
137/137 [==============================] - 34s 250ms/step - loss: 0.8357 - hinge_accuracy: 0.5956 - val_loss: 0.8621 - val_hinge_accuracy: 0.5792
Epoch 33/40
137/137 [==============================] - 34s 252ms/step - loss: 0.8328 - hinge_accuracy: 0.6009 - val_loss: 0.8642 - val_hinge_accuracy: 0.5736
Epoch 34/40
137/137 [==============================] - 34s 251ms/step - loss: 0.8304 - hinge_accuracy: 0.6020 - val_loss: 0.8493 - val_hinge_accuracy: 0.6076
Epoch 35/40
137/137 [==============================] - 34s 251ms/step - loss: 0.8295 - hinge_accuracy: 0.6019 - val_loss: 0.8386 - val_hinge_accuracy: 0.6035
Epoch 36/40
137/137 [==============================] - 34s 250ms/step - loss: 0.8282 - hinge_accuracy: 0.6021 - val_loss: 0.8391 - val_hinge_accuracy: 0.6153
Epoch 37/40
137/137 [==============================] - 34s 250ms/step - loss: 0.8262 - hinge_accuracy: 0.6037 - val_loss: 0.8326 - val_hinge_accuracy: 0.6125
Epoch 38/40
137/137 [==============================] - 34s 251ms/step - loss: 0.8240 - hinge_accuracy: 0.6059 - val_loss: 0.8248 - val_hinge_accuracy: 0.6174
Epoch 39/40
137/137 [==============================] - 34s 250ms/step - loss: 0.8226 - hinge_accuracy: 0.6058 - val_loss: 0.8155 - val_hinge_accuracy: 0.6187
Epoch 40/40
137/137 [==============================] - 34s 250ms/step - loss: 0.8203 - hinge_accuracy: 0.6080 - val_loss: 0.8096 - val_hinge_accuracy: 0.6229
45/45 [==============================] - 2s 47ms/step - loss: 0.8096 - hinge_accuracy: 0.6229
:: qnn_accuracy =  0.6229166388511658

