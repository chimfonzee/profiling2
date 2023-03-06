import torch
import torch.nn as nn

from common import load_model, evaluate, prepare_data_loaders

data_path = 'data/imagenet'
saved_model_dir = 'data/'
float_model_file = 'mobilenet_pretrained_float.pth'
scripted_float_model_file = 'mobilenet_quantization_scripted.pth'
scripted_quantized_model_file = 'mobilenet_quantization_scripted_quantized.pth'
qat_model_file = 'trained_quantized_model.pth'

train_batch_size = 30
eval_batch_size = 20
num_eval_batches = 10
criterion = nn.CrossEntropyLoss()
data_loader, data_loader_test = prepare_data_loaders(data_path, train_batch_size, eval_batch_size)

# original_model = torch.jit.load(scripted_float_model_file + float_model_file)
quantized_model = torch.jit.load(saved_model_dir + scripted_quantized_model_file)
qat_model = torch.jit.load(saved_model_dir + qat_model_file)
top1, top5 = evaluate(quantized_model, criterion, data_loader_test, neval_batches=num_eval_batches)
