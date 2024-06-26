# A simple machine learning project using scikit learn
The objective of this project is to show simple examples of how polynomial regression models works.

Features:
- Whole cycle of a Machine Learning Project in each case.
- Datasets to practice.
- Most common metrics to evaluate the model.

## Installing using GitHub
- Fork the project into your GitHub
- Clone it into your dektop
```
git clone https://github.com/jacesca/SimpleMLRegression.git
```
- Setup environment (it requires python3)
```
python -m venv venv
source venv/bin/activate  # for Unix-based system
venv\Scripts\activate  # for Windows
```
- Install requirements
```
pip install -r requirements.txt
```

## Run ML model
```
python model-height.py
python model-height-stat.py
python model-house.py
python model-height-2features.py
python model-model-house-2features.py
python model-polynomial.py
python model-house-poly.py
```

## Others
- Proyect in GitHub: https://github.com/jacesca/SimpleMLRegression
- Commands to save the environment requirements:
```
conda list -e > requirements.txt
# or
pip freeze > requirements.txt

conda env export > flask_env.yml
```
- For coding style
```
black model.py
flake8 model.py
```

## Extra documentation
- [How to plot in multiple subplots](https://stackoverflow.com/questions/31726643/how-to-plot-in-multiple-subplots)
