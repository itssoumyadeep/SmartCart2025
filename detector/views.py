import time
import hashlib
from django.shortcuts import render
from .classifier import (
    classify_vegetable,
    classify_with_gemini,
    classify_image_huggingface
)

# Simple in-memory cache
last_gemini_detection = {
    "image_hash": None,
    "veg_name": None,
    "timestamp": 0
}


def hash_image(image_base64):
    #Generate a hash of the image for comparison.
    return hashlib.md5(image_base64.encode()).hexdigest()

def should_call_llm(image_hash):
    #Check whether the LLM should be called again.
    now = time.time()
    return (
        last_gemini_detection["image_hash"] != image_hash or
        (now - last_gemini_detection["timestamp"]) > 10
    )


def index(request):
    result = None

    if request.method == "POST":
        image_data = request.POST.get("image_data")
        model_choice = request.POST.get("model_name")  # get model from form

        print("Image data received:", bool(image_data))
        print("Model selected:", model_choice)

        if image_data:
            veg_name = "Unknown"
            image_hash = hash_image(image_data)

            if model_choice == "mobilenet":
                veg_name = classify_vegetable(image_data)

            elif model_choice == "huggingface_vit":
                veg_name = classify_image_huggingface(image_data)

            elif model_choice == "gemini":
                if should_call_llm(image_hash):
                    veg_name = classify_with_gemini(image_data)
                    last_gemini_detection.update({
                        "image_hash": image_hash,
                        "veg_name": veg_name,
                        "timestamp": time.time()
                    })
                else:
                    print("[INFO] Using cached Gemini result.")
                    veg_name = last_gemini_detection["veg_name"]

            result = {
                "items": [{"name": veg_name, "price": estimate_price(veg_name)}],
                "total": estimate_price(veg_name)
            }

    return render(request, "index.html", {"result": result})

# Basic price estimation
def estimate_price(name):
    price_list = {
        "broccoli": 1.5,
        "carrot": 0.8,
        "cabbage": 1.2,
        "tomato": 1.0,
        "cucumber": 0.9,
        "eggplant": 2.99,
        "beetroot": 3.99,
        "cauliflower": 2.99,
        "potato": 0.5,
    }
    for key in price_list:
        if key in name.lower():
            return price_list[key]
    return 0.0  # default
