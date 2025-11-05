from cog import BasePredictor, Input
import torch

class Predictor(BasePredictor):
    def setup(self):
        # Ladda modellen
        self.model = torch.load("checkpoints/planritningsmodel.ckpt")

    def predict(self, image: Input(description="Input image")):
        # Gör prediktion med modellen
        return image  # ersätt med riktig output

