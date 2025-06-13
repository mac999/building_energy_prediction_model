# Title: DCE MLP program
# Author: Kang Taewook
# Created Date: 2025.5.19
# Description: DCE prediction program daily
#
import os, argparse, json, torch, torch.nn as nn, pandas as pd, numpy as np
from sklearn.preprocessing import MinMaxScaler

# ERC model definition
class DCE_MLP(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, dropout_ratio=0.0):
		super(DCE_MLP, self).__init__()
		layers = []
		layers.append(nn.Linear(input_size, hidden_size[0]))
		layers.append(nn.ReLU())
		layers.append(nn.BatchNorm1d(hidden_size[0]))
		layers.append(nn.Dropout(dropout_ratio))
		for i in range(1, len(hidden_size)):
			layers.append(nn.Linear(hidden_size[i-1], hidden_size[i]))
			layers.append(nn.ReLU())
			layers.append(nn.BatchNorm1d(hidden_size[i]))
			layers.append(nn.Dropout(dropout_ratio))
		layers.append(nn.Linear(hidden_size[-1], output_size))
		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		return self.layers(x)

# load config file
def load_model_from_config(config_path, model_file_override=None):
	with open(config_path, 'r') as f:
		config = json.load(f)
	model_type = config.get('model', 'mlp')
	input_size = len(config['input_cols'])
	hidden_size = config['hidden_size']
	output_size = 1
	dropout_ratio = config.get('dropout_ratio', 0.0)
	model_fname = model_file_override if model_file_override else config['model_fname']

	if model_type == 'mlp':
		model = DCE_MLP(input_size, hidden_size, output_size, dropout_ratio)
	else:
		raise NotImplementedError(f"Not support model type: {model_type}")

	model_file = torch.load(model_fname, map_location='cpu')
	model.load_state_dict(model_file)
	model.eval()
	return model, config

# fitting the dataset for using the trained model
def fit_scaler_from_csv(csv_path, input_cols, config):
	df = pd.read_csv(csv_path)
	# remove NaT, NULL row 
	rows_to_drop = []
	for row in range(df.shape[0]):
		remove = False
		for col in range(df.shape[1]):
			column_name = df.columns[col]
			if column_name not in input_cols + ['EnergyConsumption']:
				continue
			if pd.isnull(df.iat[row, col]) or pd.isna(df.iat[row, col]):
				remove = True
				break
		if remove:
			rows_to_drop.append(row)
	if rows_to_drop:
		df = df.drop(rows_to_drop, axis=0).reset_index(drop=True)

	input_cols = input_cols + ['EnergyConsumption']  
	if config is not None and 'filter_energy' in config:
		ds = df[input_cols].values
		ds = ds[ds[:, len(input_cols)-1] > config['filter_energy']]
		df = pd.DataFrame(ds, columns=input_cols)
			
	# input_cols = input_cols[:-1] 
	min_list = config['energy_usage_minmax_data_min']
	max_list = config['energy_usage_minmax_data_max']
	input_dataset = np.array([min_list, max_list])

	scaler_ds = MinMaxScaler()
	scaler_ds.fit(input_dataset)
	# scaler_ds.min_[-1], scaler_ds.scale_[-1] = scales
	# scaler_ds.data_min_[-1], scaler_ds.data_max_[-1] = minmax 
	# scaler.fit(df[fit_cols].values)
	return scaler_ds

# predict data
def predict_on_csv(model, scaler, input_cols, csv_path):
	df = pd.read_csv(csv_path)
	inputs = df[input_cols].values
	inputs_scaled = scaler.transform(np.concatenate([inputs, np.zeros((inputs.shape[0], 1))], axis=1))[:, :-1]
	inputs_tensor = torch.tensor(inputs_scaled, dtype=torch.float32)
	with torch.no_grad():
		preds = model(inputs_tensor).cpu().numpy()

	dummy = np.zeros((preds.shape[0], len(input_cols)+1))
	dummy[:, :-1] = inputs
	dummy[:, -1] = preds.flatten()
	preds_real = scaler.inverse_transform(dummy)[:, -1]
	df['Predicted_EnergyConsumption'] = preds_real
	return df

def batch_predict(model, scaler, input_cols, input_folder, output_folder):
	csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
	if not csv_files:
		print("no input file.")
		return []
	os.makedirs(output_folder, exist_ok=True)
	output_files = []
	for csv_file in csv_files:
		csv_path = os.path.join(input_folder, csv_file)
		print(f"Predicting {csv_file} ...")
		df_pred = predict_on_csv(model, scaler, input_cols, csv_path)
		out_path = os.path.join(output_folder, csv_file)
		df_pred.to_csv(out_path, index=False)
		print(f"Saved: {out_path}")
		output_files.append(out_path)
	return output_files

# CLI or gradio web app
def run_cli(args):
	model, config = load_model_from_config(args.model_file)
	input_cols = config['input_cols']
	csv_files = [f for f in os.listdir(args.input_folder) if f.endswith('.csv')]
	if not csv_files:
		print("no input csv file.")
		return
	first_df_path = os.path.join(args.input_folder, csv_files[0])
	scaler = fit_scaler_from_csv(first_df_path, input_cols, config)
	batch_predict(model, scaler, input_cols, args.input_folder, args.output_folder)

def run_gradio():
	import gradio as gr
	import tempfile
	import shutil
	import zipfile

	def predict_gradio(config_json, model_file, csv_files):
		with tempfile.TemporaryDirectory() as tmpdir:
			config_path = os.path.join(tmpdir, "config.json")
			model_path = os.path.join(tmpdir, "model.pth")
			input_folder = os.path.join(tmpdir, "input")
			output_folder = os.path.join(tmpdir, "output")
			os.makedirs(input_folder, exist_ok=True)
			os.makedirs(output_folder, exist_ok=True)
			config_json.save(config_path)
			model_file.save(model_path)
			input_csv_paths = []
			for csv in csv_files:
				csv_path = os.path.join(input_folder, csv.name)
				csv.save(csv_path)
				input_csv_paths.append(csv_path)
			model, config = load_model_from_config(config_path, model_file_override=model_path)

			input_cols = config['input_cols']
			scaler = fit_scaler_from_csv(input_csv_paths[0], input_cols)
			batch_predict(model, scaler, input_cols, input_folder, output_folder)
			zip_path = os.path.join(tmpdir, "predicted_csv.zip")
			with zipfile.ZipFile(zip_path, 'w') as zipf:
				for file in os.listdir(output_folder):
					zipf.write(os.path.join(output_folder, file), arcname=file)
			return zip_path

	demo = gr.Interface(
		fn=predict_gradio,

		inputs=[
			gr.File(label="model_config.json", file_types=[".json"], height=65),
			gr.File(label="Trained Model File (.pth)", file_types=[".pth"], height=65),
			gr.Files(label="Input CSV Files", file_types=[".csv"], height=67)
		],
		outputs=gr.File(label="Predicted CSVs (ZIP)"),
		title="KICT Energy Prediction Model",
		description="Upload model_config.json, the trained model file (.pth), and input CSV files. Click Predict to download the prediction results as a ZIP file."
	)
	demo.launch(share=True)

def main():
	parser = argparse.ArgumentParser() 
	parser.add_argument('--model_file', required=False, default='./output/20250601-2220/dce_predict_model.json', help='dce model config.json file path')
	parser.add_argument('--input_folder', required=False, default='./predict_input', help='input model path')
	parser.add_argument('--output_folder', required=False, default='./predict_output', help='predict output path')
	parser.add_argument('--webapp_server', required=False, default=True, help='webapp server mode')
	args = parser.parse_args()

	if args.webapp_server:
		run_gradio()
	else:
		run_cli(args)

if __name__ == '__main__':
	main()