from django.shortcuts import render
from PIL import Image
import base64
from io import BytesIO
import requests

# 🥦 1. Define your vegetable-price mapping
VEGETABLE_PRICES = {
    "Cucumber": 1.50,
    "Mango": 2.00,
    "Tomato": 1.20,
    "Carrot": 0.80,
    "Onion": 1.70,
    "Potato": 0.9,
    "Eggplant": 3.1,
    "Yam": 2.05,
    "Apple": 2.50,
    "lemon": 1.05
}

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llava"  # or your preferred model

# 🔍 2. Decode image from base64
def decode_image_from_base64(base64_str):
    base64_str = base64_str.split(',')[1]  # Remove data:image/jpeg;base64,
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))
    return image

# 🧠 3. Detect vegetable names using Ollama
def detect_vegetables(image):
    image = image.convert("RGB")
    image.thumbnail((512, 512))

    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    prompt = (
        "Which of the following vegetables are in this image: "
        f"{', '.join(VEGETABLE_PRICES.keys())}? "
        "Only reply with a comma-separated list of those that are visible. "
        "If none are found, reply with 'None'."
    )

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)
    return response.json().get("response", "").strip()

# 🎯 4. Main Django view
def index(request):
    result = None
    detected_items = []

    if request.method == 'POST':
        image_data = request.POST.get('image_data')
        if image_data:
            image = decode_image_from_base64(image_data)
            response_text = detect_vegetables(image)

            if response_text.lower() != "none":
                # Parse the response into a list of vegetables
                found = [item.strip() for item in response_text.split(',') if item.strip() in VEGETABLE_PRICES]

                # Prepare data for template
                detected_items = [
                    {"name": veg, "price": VEGETABLE_PRICES[veg]}
                    for veg in found
                ]

                total_price = sum(item["price"] for item in detected_items)
                result = {"items": detected_items, "total": total_price}
            else:
                result = {"items": [], "total": 0}

    return render(request, 'index.html', {'result': result})
