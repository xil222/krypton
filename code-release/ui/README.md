#Steps for developing on mac

1. Intstall virtualenv by pip install virtualenv`
2. Install django `pip install django`
3. Create a virtual environment at this directory level `python -m vitualenv venv_django`
4. Activate the virtualenv `source venv_django/bin/activate`
5. Install all dependencies `pip install requirements.txt`
6. Run python manage.py makemigrations
7. Run python manage.py migrate

## To run the environment
1. cd to the krypton_ui where manage.py is located.
2. run the following `python manage.py runserver`
3. open the browser at localhost

#Steps for development on kubectl pod

1. apt-get update
2. Install git, django, wget, opencv
   apt-get install git
   apt-get install python-django
   apt-get install wget
   apt-get install python-opencv
3. git clone https://github.com/scnakandala/krypton.git
4. cd /krypton/code-release/core make
5. cd /krypton/code-release  run ./download_cnn_weights.sh

## To run the environment
steps 1 same
2. run the server with a specific part for e.g python manage.py runserver 8080
3. open another cmd, go to kubectl folder and make port forward run  kubectl port-forward krypton 8080
