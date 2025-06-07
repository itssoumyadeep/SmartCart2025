import base64
import io
import os
from PIL import Image
import torch
from torchvision import models, transforms
import google.generativeai as genai

# -------------------- Gemini Vision -------------------- #
genai.configure(api_key=os.getenv("GEMINI_API_KEY", "XXX"))

def classify_with_gemini(image_base64):
    image_data = base64.b64decode(image_base64.split(',')[1])
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    #model = genai.GenerativeModel('gemini-pro-vision')
    model = genai.GenerativeModel('gemini-1.5-pro')
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = "What vegetable is shown in this image? Just reply with the name."

    #response = model.generate_content([prompt, image])
    response = model.generate_content(
        [prompt, image],
        generation_config={"temperature": 0.2}
    )
    return response.text.strip()

# -------------------- MobileNetV2 Custom Model -------------------- #
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Load model
mobilenet_model = models.mobilenet_v2(weights=None)
mobilenet_model.classifier[1] = torch.nn.Linear(mobilenet_model.last_channel, 36)
mobilenet_model.load_state_dict(torch.load("vegetable_mobilenet.pth", map_location=torch.device('cpu')))
mobilenet_model.eval()

# Load labels
with open("vegetable_labels.txt") as f:
    labels = [line.strip() for line in f.readlines()]

def classify_vegetable(image_base64):
    image_data = base64.b64decode(image_base64.split(',')[1])
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = mobilenet_model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        predicted_index = probs.argmax(1).item()
        predicted_label = labels[predicted_index]
        confidence = probs[0][predicted_index].item()

        print(f"[MobileNet] {predicted_label} (conf: {confidence:.2f})")
        print(f"Top 5: {[(labels[i], float(probs[0][i])) for i in probs[0].argsort(descending=True)[:5]]}")

    return predicted_label

# -------------------- Hugging Face Placeholder -------------------- #
# Optional future support
def classify_image_huggingface(image_base64):
    # TODO: Integrate actual Hugging Face model here
    return "huggingface-classifier-not-implemented"
