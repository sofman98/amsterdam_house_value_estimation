# Amsterdam House Price Estimation

Build the docker image with the command:
```
$ docker build -t woz_estimation .
```

Run the docker image and specify which port to use:
```
$ docker run -p {your_port}:80 woz_estimation
```

This will train and evaluate a linear regression model, then will deploy an api accessible at the port you provided.

You can also independently train and evalute the model by running:
```
$ python -m src.scripts.train_and_evaluate_model
```
The metric scores will be found inside ```assets/metrics/```.

### Example of an endpoint:
http://localhost:80/api/get_woz_value?single=543&married_no_kids=37&not_married_no_kids=149&married_with_kids=14&not_married_with_kids=12&single_parent=22&other=12