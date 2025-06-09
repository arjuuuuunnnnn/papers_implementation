## PyTorch implementation of Attention is all you need paper

[Paper](https://arxiv.org/abs/1706.03762)

### Usage
1. Change the `load_data` function in train.py to load your own dataset, use any tokenizer you prefer (I have used a simple white-space tokenizer for simplicity).

2. Run the training script:
```bash
python train.py
```
3. The model will be saved as `transformer.pt` after training, run the inference script to test the model:
```bash
python inference.py
```

