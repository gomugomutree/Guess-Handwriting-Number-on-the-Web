from torch import load

def load_model():
    model = load('models/model.epoch-20.bs-512.layer-5.ratio-0.8.lowest_loss-0.001.pth')
    return model

