from django.shortcuts import render, redirect 
from django.http import HttpResponse
from django.forms import inlineformset_factory
from django.contrib.auth.forms import UserCreationForm
from .models import otherDetails
from django.contrib.auth import authenticate, login, logout
from .forms import img
from django.contrib import messages

from django.contrib.auth.decorators import login_required
from .models import *
from .forms import CreateUserForm
# Create your views here.


def loginPage(request):
    if request.user.is_authenticated:
        return redirect('')
    else:
        print(request.method)
        if request.method == 'POST':
            if 'login' in request.POST:
                username = request.POST.get('username')
                password = request.POST.get('password')

                user = authenticate(request, username=username, password=password)

                if user is not None:
                    login(request, user)
                    
                    return redirect('')
                else:
                    messages.info(request, 'Username OR password is incorrect')
            elif 'register' in request.POST:
                if request.user.is_authenticated:
                    return redirect('login')
                else:
                    form = CreateUserForm()
                    if request.method == 'POST':
                        username = request.POST.get('username')
                        password = request.POST.get('password')
                        prodName = request.POST.get('prodName')
                        
                        user = User.objects.create_user(username=username, password=password)
                        # user.userprofile.user = authenticate(username=username, password=password)
                        user.save()
                        user = authenticate(username=username, password=password)
                        
                        profile = Profile()
                        profile.user = user
                        profile.prodName = prodName
                        profile.save()

                        login(request, user)
                        messages.success(request, 'Account was created')
                        print("Account was created")


                        return redirect('login')

        return render(request, 'index.html')
            

def logoutUser(request):
	logout(request)
	return redirect('login')

def bulk(request):
    if request.method == "POST":
        my_file = request.FILES.get("file")
        otherDetails.objects.create(image = my_file)
        return redirect("/bulk")
    else:
        form = img()
        return render(request, 'bulk.html')
def main(request):
    return HttpResponse("hello")