import base64

# Example keys and values from your output (encoded in Base64)
example_data = [
    {
        "key": "c3BlbmRlcg==",
        "value": "c2VpMW54Njl5eWFzOXVoMmVxZmNlMGdueHU3dTlqcmV4dnBscXltOHps",
    },
    {"key": "YW1vdW50", "value": "NzkzNjh1c2Vp"},
    {"key": "YWN0aW9u", "value": "L2Nvc213YXNtLndhc20udjEuTXNnRXhlY3V0ZUNvbnRyYWN0"},
    {"key": "Y29kZV9pZA==", "value": "NDc4"},
    {"key": "bW9kdWxl", "value": "d2FzbQ=="},
    {
        "key": "cGFpcg==",
        "value": "c2VpMTVrOG01bnd3c2xxdWFtZTQzaDY2anQydWpzajJsbXFmMnNxM3NhNTdzbTYyMmh5NjdsNHFucTV1ajktdXNlaQ==",
    },
]

# Decoding
for data in example_data:
    decoded_key = base64.b64decode(data["key"]).decode()
    decoded_value = base64.b64decode(data["value"]).decode()
    print(f"Decoded Key: {decoded_key}, Decoded Value: {decoded_value}")
