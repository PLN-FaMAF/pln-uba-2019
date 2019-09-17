# pln-uba-2019
Introducción al Procesamiento de Lenguaje Natural - UBA 2019


## Instalación

1. Se necesita el siguiente software:

   - Git
   - Pip
   - Python 3.4 o posterior
   - Virtualenvwrapper

   En un sistema basado en Debian (como Ubuntu), se puede hacer::

    sudo apt-get install git python-pip python3 virtualenvwrapper

2. Crear y activar un nuevo
   `virtualenv <http://virtualenv.readthedocs.org/en/latest/virtualenv.html>`_::

    mkvirtualenv --python=/usr/bin/python3 pln

3. Bajar el código::

    git clone https://github.com/PLN-FaMAF/PLN-2019.git

4. Instalarlo::

    cd PLN-2019
    pip install -r requirements.txt


## Ejecución

1. Activar el entorno virtual con::

    workon pln

2. Correr el script que uno quiera. Por ejemplo::

    python tagging/scripts/train.py -h


## Testing

Correr nose::

    nosetests


## Jupyter (IDE y notebooks)

Correr Jupyter Lab (abre una ventana en el navegador por defecto)::

    jupyter lab
