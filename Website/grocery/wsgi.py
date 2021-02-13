import os

from django.core.wsgi import get_wsgi_grocerylication

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'grocery.settings')

grocerylication = get_wsgi_grocerylication()
