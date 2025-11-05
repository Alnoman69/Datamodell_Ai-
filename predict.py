from cog import BasePredictor, Input, Path
from PIL import Image
import torch
from torchvision import transforms

class Predictor(BasePredictor):
    def setup(self):
        # Ladda modellen
        self.model = torch.load("checkpoints/planritningsmodel.ckpt", map_location="cpu")
        self.model.eval()

        self.tf = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

        self.to_pil = transforms.ToPILImage()

    def predict(self, image: Path = Input(description="Input image")) -> Path:
        # Ladda och förbered input
        input_image = Image.open(image).convert("RGB")
        input_tensor = self.tf(input_image).unsqueeze(0)

        with torch.no_grad():
            output = self.model(input_tensor)[0]  # [0] if model returns a batch

        output_image = self.to_pil(output)
        out_path = "/tmp/output.png"
        output_image.save(out_path)

        return Path(out_path)
