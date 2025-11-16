import requests
import base64
from io import BytesIO
from PIL import Image
import cv2
import os
from openai import OpenAI


def img2b64(img: Image):
    buffered = BytesIO()
    img = img.convert('RGB')
    img.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()
    encoded = base64.b64encode(img_bytes).decode('ascii')
    return encoded

def ima2b64_cv2(img):
    _, buffer = cv2.imencode('.jpg', img)
    img_bytes = buffer.tobytes()
    encoded = base64.b64encode(img_bytes).decode('utf-8')
    return encoded



class Agent(object):
    def __init__(self, api_key: str, base_url: str, instruct: str, model: str):
        self.api_key = api_key 
        self.base_url = base_url 
        self.model = model  
        # self.max_tokens = max_tokens
        self.instruct = instruct
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def chat_vlm(self, image, text: str):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.instruct}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64," + ima2b64_cv2(image)
                            },
                        },
                        {"type": "text", "text": text},
                    ],
                },
            ],
            # temperature=0.2     
            # max_tokens=self.max_tokens
        )

        return completion.choices[0].message.content

    def chat_vlm2json(self, image, text: str):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.instruct}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64," + ima2b64_cv2(image)
                            },
                        },
                        {"type": "text", "text": text},
                    ],
                },
            ],
            response_format={"type": "json_object"},
            # temperature=0.2  
            # max_tokens=self.max_tokens
        )
        return completion.choices[0].message.content

    def chat2json(self, text: str):
        completion = self.client.chat.completions.create(
            model=self.model, 
            messages=[
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.instruct}],
                },
                {
                    "role": "user",
                    "content": text, 
                },
            ],
            response_format={"type": "json_object"},
            # temperature=0.2  
            # max_tokens=self.max_tokens
        )
        return completion.choices[0].message.content

    def chat(self, text: str):
        completion = self.client.chat.completions.create(
        model=self.model,
        messages=[
            {"role": "system", "content":  [{"type": "text", "text": self.instruct}]},
            {"role": "user", "content": text},
        ],
        # temperature=0.2
        # max_tokens=self.max_tokens
        )
        return completion.choices[0].message.content


    @classmethod
    def from_config(cls, config, instruct):
        return cls(
            api_key=config.get("api_key"),
            base_url=config.get("base_url"),
            model=config.get("model"),
            # max_tokens=config.get("max_tokens"),
            instruct=instruct
        )

