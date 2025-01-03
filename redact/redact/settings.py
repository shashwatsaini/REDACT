"""
Django settings for redact project.

Generated by 'django-admin startproject' using Django 5.1.1.

For more information on this file, see
https://docs.djangoproject.com/en/5.1/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/5.1/ref/settings/
"""

from pathlib import Path

import os
import json

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/5.1/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-q#rx5f$rkpvvn&lzf=55%=2#)ppvcyu954gd6n*8!fzj8k=r33'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = []


# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'app'
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'redact.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'redact.wsgi.application'


# Database
# https://docs.djangoproject.com/en/5.1/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}


# Password validation
# https://docs.djangoproject.com/en/5.1/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/5.1/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/5.1/howto/static-files/

STATIC_URL = 'static/'

# Default primary key field type
# https://docs.djangoproject.com/en/5.1/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

Ollama_API_URL = 'http://localhost:11434/v1'
Ollama_API_KEY = 'ollama'

with open(os.path.join(BASE_DIR, 'app', 'services', 'service_keys.json')) as f:
    json_data = json.load(f)
    AZURE_DI_KEY = json_data['AZURE_DI_KEY']
    AZURE_DI_ENDPOINT = json_data['AZURE_DI_ENDPOINT']
    AZURE_CS_KEY = json_data['AZURE_CS_KEY']
    AZURE_CS_ENDPOINT = json_data['AZURE_CS_ENDPOINT']
    AZURE_SI_KEY = json_data['AZURE_SI_KEY']
    AZURE_SI_ENDPOINT = json_data['AZURE_SI_ENDPOINT']
    AZURE_STORAGE_URL = json_data['AZURE_STORAGE_URL']
    AZURE_STORAGE_NAME = json_data['AZURE_STORAGE_NAME']
    AZURE_STORAGE_CONTAINER = json_data['AZURE_STORAGE_CONTAINER']
    AZURE_VI_NAME = json_data['AZURE_VI_NAME']
    AZURE_VI_ID = json_data['AZURE_VI_ID']
    AZURE_VI_RESOURCE_GROUP = json_data['AZURE_VI_RESOURCE_GROUP']
    AZURE_VI_RESOURCE_MANAGER = json_data['AZURE_VI_RESOURCE_MANAGER']
    AZURE_VI_LOCATION = json_data['AZURE_VI_LOCATION']
    AZURE_VI_SUBSCRIPTION = json_data['AZURE_VI_SUBSCRIPTION']
    AZURE_VI_API_ENDPOINT = json_data['AZURE_VI_API_ENDPOINT']

MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

MODEL_PATH = os.path.join(BASE_DIR, 'models')
MODEL_TRAINING_LOGS = os.path.join(MODEL_PATH, 'training_logs.json')

YOLO_MODEL_ROOT = os.path.join(BASE_DIR, 'yolo')
YOLO_MODEL_PATH = os.path.join(BASE_DIR, 'yolo', 'yolov8n_100e.pt')
