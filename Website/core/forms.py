from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django import forms
from django import forms
from core.models import otherDetails
from .models import Profile

class CreateUserForm(UserCreationForm):
    prodName = forms.CharField()
    class Meta:
        model = User
        fields = [ 'username', 'prodName', 'password']

class img(forms.ModelForm):
    class Meta:
        model=otherDetails
        fields = "__all__"
        image = forms.FileField(widget=forms.ClearableFileInput(attrs={'multiple': True}))

