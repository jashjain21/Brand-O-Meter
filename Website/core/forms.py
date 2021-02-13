from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django import forms

from .models import Profile

class CreateUserForm(UserCreationForm):
    prodName = forms.CharField()
    class Meta:
        model = User
        fields = [ 'username', 'prodName', 'password']
