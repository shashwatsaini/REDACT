import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import AnalyzeTextOptions, AnalyzeImageOptions, TextCategory, ImageData, ImageCategory
from django.conf import settings
import regex as re

import nltk
nltk.download('averaged_perceptron_tagger_eng')

# Guardrail for Azure content safety
def guardrail_azure_cs_text(word_list):
    flag = 0
    text = ' '.join(word_list)
    endpoint = settings.AZURE_CS_ENDPOINT
    key = settings.AZURE_CS_KEY

    content_safety_client = ContentSafetyClient(
        endpoint=endpoint, credential=AzureKeyCredential(key)
    )

    request = AnalyzeTextOptions(text=text)
    response = content_safety_client.analyze_text(request)

    hate_result = next(item for item in response.categories_analysis if item.category == TextCategory.HATE)
    self_harm_result = next(item for item in response.categories_analysis if item.category == TextCategory.SELF_HARM)
    sexual_result = next(item for item in response.categories_analysis if item.category == TextCategory.SEXUAL)
    violence_result = next(item for item in response.categories_analysis if item.category == TextCategory.VIOLENCE)

    if hate_result and hate_result.severity >= 4:
        flag = 1
    if self_harm_result and self_harm_result.severity >= 4:
        flag = 1
    if sexual_result and sexual_result.severity >=  4:
        flag = 1
    if violence_result and violence_result.severity >= 4:
        flag = 1

    return flag

def guardrail_azure_cs_image(image):
    flag = 0
    endpoint = settings.AZURE_CS_ENDPOINT
    key = settings.AZURE_CS_KEY

    content_safety_client = ContentSafetyClient(
        endpoint=endpoint, credential=AzureKeyCredential(key)
    )

    with open(image, 'rb') as file:
        request = AnalyzeImageOptions(image=ImageData(content=file.read()))
        response = content_safety_client.analyze_image(request)

    hate_result = next(item for item in response.categories_analysis if item.category == ImageCategory.HATE)
    self_harm_result = next(item for item in response.categories_analysis if item.category == ImageCategory.SELF_HARM)
    sexual_result = next(item for item in response.categories_analysis if item.category == ImageCategory.SEXUAL)
    violence_result = next(item for item in response.categories_analysis if item.category == ImageCategory.VIOLENCE)

    if hate_result and hate_result.severity >= 4:
        flag = 1
    if self_harm_result and self_harm_result.severity >= 4:
        flag = 1
    if sexual_result and sexual_result.severity >=  4:
        flag = 1
    if violence_result and violence_result.severity >= 4:
        flag = 1

    return flag

# Guardrail that redacts proper nouns
def guardrail_proper_nouns(word_list):
    from nltk.tag import pos_tag
    
    redacted_list = []
    for word in word_list:
        word = word.strip()
        if word:
            tag = pos_tag([word])[0][1]
            if tag == 'NNP':
                redacted_list.append(word)

    return list(set(redacted_list))

# Guardrail that redacts numbers
def guardrail_numbers(word_list):
    patterns = {
        "numbers": re.compile(r"\b\d+[\d.,-]*\b"),
    }

    redacted_list = []
    for pattern in patterns.values():
        matches = pattern.findall(' '.join(word_list))
        for match in matches:
            match_text = ''.join(match).strip()
            if match_text:
                redacted_list.append(match_text)
    
    return redacted_list

# Guardrail that redacts URLs
def guardrail_urls(word_list):
    patterns = {
        "urls": re.compile(r'(?:https?://)?(?:www\.)?([\w.-]+\.\w+)'),
    }

    redacted_list = []
    for pattern in patterns.values():
        matches = pattern.findall(' '.join(word_list))
        for match in matches:
            match_text = ''.join(match).strip()
            if match_text:
                redacted_list.append(match_text)
    
    return redacted_list

# Guardrail that redacts emails
def guardrail_emails(word_list):
    patterns = {
        "email": re.compile(r"(?<!\*)\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b(?!\*)"),
        "obfuscated_email": re.compile(r"(?<!\*)[A-Za-z0-9._%+-]+\s?at\s?[A-Za-z0-9.-]+\s?dot\s?[A-Za-z]{2,}(?!\*)"),
    }

    redacted_list = []
    for pattern in patterns.values():
        matches = pattern.findall(' '.join(word_list))
        for match in matches:
            match_text = ''.join(match).strip()
            if match_text:
                redacted_list.append(match_text)
    
    return redacted_list
