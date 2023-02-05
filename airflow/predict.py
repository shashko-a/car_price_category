import pickle
import os
import json
import pandas as pd


path = os.path.expanduser('~/airflow')

def predict():
	predict = {}
    
	dirname_model = f'{path}/data/models/'
	list_model = os.listdir(dirname_model)
	model_name = f'{dirname_model}{list_model[0]}'
	
	with open (model_name, "rb") as file:
		model = pickle.load(file)

	dirname_json = f'{path}/data/test/'
	list_json = os.listdir(dirname_json)
	
	for i in(list_json):
	    df = pd.read_json(f'{dirname_json}{i}', orient='index').T
	    
	    predict[f'{i}']=model.predict(df)
	    
	df2 = pd.DataFrame.from_dict(predict, orient='index')
	df2.to_csv(f'{path}/data/predictions/preds.csv')


if __name__ == '__main__':
    predict()
