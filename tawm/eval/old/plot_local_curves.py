import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(path: str):
	os.makedirs(path, exist_ok=True)


def plot_disassemble_eval_curves():
	# Try typical filenames produced by eval_model_multidt.py
	candidates = [
		('timeaware', 'tawm/logs/mw-disassemble/eval_multidt_timeaware_euler_4.csv'),
		('baseline', 'tawm/logs/mw-disassemble/eval_multidt_baseline_0.0025_4.csv'),
	]

	frames = {'timeaware': [], 'baseline': []}
	for label, path in candidates:
		if os.path.exists(path):
			frames[label].append(pd.read_csv(path))

	if not frames['timeaware'] or not frames['baseline']:
		print('Missing CSVs for mw-disassemble. Skipping eval curves.')
		return

	df_tawm = pd.concat(frames['timeaware'], ignore_index=True)
	df_base = pd.concat(frames['baseline'], ignore_index=True)

	metric = 'Success'
	agg_tawm = df_tawm.groupby('Timestep').agg(avg=(metric, 'mean')).reset_index()
	agg_base = df_base.groupby('Timestep').agg(avg=(metric, 'mean')).reset_index()

	agg_tawm['Timestep'] = agg_tawm['Timestep'] * 1000.0
	agg_base['Timestep'] = agg_base['Timestep'] * 1000.0

	ensure_dir('tawm/plots')
	plt.figure(figsize=(8,5))
	plt.title('mw-disassemble: Success vs Δt', fontsize=18)
	plt.plot(agg_base['Timestep'], agg_base['avg'], label='Baseline', color='blue', linewidth=3)
	plt.plot(agg_tawm['Timestep'], agg_tawm['avg'], label='Time-aware (Euler)', color='red', linewidth=3)
	plt.xlabel('Δt (ms)')
	plt.ylabel('Success Rate')
	plt.grid(True)
	plt.legend()
	plt.savefig('tawm/plots/timeaware-mw-disassemble.png', bbox_inches='tight', pad_inches=0.1)
	plt.close()


def plot_basketball_learning_curves():
	# Use training curve files (return_curve_*.csv)
	base_csv = 'tawm/logs/mw-basketball/3/default/eval-baseline-0.0025.csv'
	tawm_csv = 'tawm/logs/mw-basketball/3/default/eval-timeaware-log-uniform.csv'
	outdir = 'tawm/plots'
	ensure_dir(outdir)

	if not (os.path.exists(base_csv) and os.path.exists(tawm_csv)):
		print('Missing training CSVs for mw-basketball. Skipping.')
		return

	df_base = pd.read_csv(base_csv)
	df_tawm = pd.read_csv(tawm_csv)

	metric = 'episode_success'
	# Round steps to nearest 1000 for alignment
	for df in [df_base, df_tawm]:
		df['step'] = (df['step'] - (df['step'] % 1000)).astype(int)

	# Aggregate by step
	eval_base = df_base.groupby('step').agg(avg=(metric, 'mean')).reset_index()
	eval_tawm = df_tawm.groupby('step').agg(avg=(metric, 'mean')).reset_index()

	# Align max common step
	n_steps = min(eval_base['step'].max(), eval_tawm['step'].max())
	eval_base = eval_base[eval_base['step'] <= n_steps]
	eval_tawm = eval_tawm[eval_tawm['step'] <= n_steps]

	plt.figure(figsize=(8,5))
	plt.title('mw-basketball', fontsize=18)
	plt.plot(eval_base['step'], eval_base['avg'], label='Baseline (dt=2.5ms)', color='blue', linewidth=3)
	plt.plot(eval_tawm['step'], eval_tawm['avg'], label='Time-aware (log-uniform)', color='red', linewidth=3)
	plt.xlabel('Steps')
	plt.ylabel('Success Rate')
	plt.grid(True)
	plt.legend()
	plt.savefig(os.path.join(outdir, 'return_curve_mw-basketball.png'), bbox_inches='tight', pad_inches=0.1)
	plt.close()


def main():
	plot_basketball_learning_curves()  # Training curves for basketball
	plot_disassemble_eval_curves()     # Eval curves for disassemble


if __name__ == '__main__':
	main()
