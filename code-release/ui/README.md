#Steps for developing

1. Intstall virtualenv by `pip install virtualenv`
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
