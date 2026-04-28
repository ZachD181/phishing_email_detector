import json
from lambda_function import lambda_handler

event = {
    "body": json.dumps({
        "email_text": "Your account has been locked. Verify immediately."
    })
}

response = lambda_handler(event, None)

print("Status Code:", response["statusCode"])
print("Body:")
print(response["body"])