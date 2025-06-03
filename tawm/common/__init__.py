MODEL_SIZE = { # parameters (M)
	1:   {'enc_dim': 256,
		  'mlp_dim': 384,
		  'latent_dim': 128,
		  'num_enc_layers': 2,
		  'num_q': 2},
	5:   {'enc_dim': 256,
		  'mlp_dim': 512,
		  'latent_dim': 512,
		  'num_enc_layers': 2},
	19:  {'enc_dim': 1024,
		  'mlp_dim': 1024,
		  'latent_dim': 768,
		  'num_enc_layers': 3},
	48:  {'enc_dim': 1792,
		  'mlp_dim': 1792,
		  'latent_dim': 768,
		  'num_enc_layers': 4},
	317: {'enc_dim': 4096,
		  'mlp_dim': 4096,
		  'latent_dim': 1376,
		  'num_enc_layers': 5,
		  'num_q': 8},
}

TASK_SET = {
	'mt9': [
		# 9 meta-world tasks
		'mw-assembly', 'mw-basketball', 'mw-box-open', 
        'mw-coffee-button', 'mw-faucet-open', 'mw-handle-pull',
        'mw-lever-pull', 'mw-soccer', 'mw-window-close',
	],
	'mt10': [
		# 10 meta-world tasks
		'mw-assembly', 'mw-basketball', 'mw-box-open', 
        'mw-coffee-button', 'mw-faucet-open', 'mw-handle-pull',
        'mw-lever-pull', 'mw-stick-pull', 'mw-soccer', 'mw-window-close',
	],
	'mt50': [
		# meta-world mt50
		'mw-assembly', 'mw-basketball', 'mw-button-press-topdown', 'mw-button-press-topdown-wall', 'mw-button-press',
		'mw-button-press-wall', 'mw-coffee-button', 'mw-coffee-pull', 'mw-coffee-push', 'mw-dial-turn',
		'mw-disassemble', 'mw-door-open', 'mw-door-close', 'mw-drawer-close', 'mw-drawer-open',
		'mw-faucet-open', 'mw-faucet-close', 'mw-hammer', 'mw-handle-press-side', 'mw-handle-press',
		'mw-handle-pull-side', 'mw-handle-pull', 'mw-lever-pull', 'mw-peg-insert-side', 'mw-peg-unplug-side',
		'mw-pick-out-of-hole', 'mw-pick-place', 'mw-pick-place-wall', 'mw-plate-slide', 'mw-plate-slide-side',
		'mw-plate-slide-back', 'mw-plate-slide-back-side', 'mw-push-back', 'mw-push', 'mw-push-wall',
		'mw-reach', 'mw-reach-wall', 'mw-shelf-place', 'mw-soccer', 'mw-stick-push',
		'mw-stick-pull', 'mw-sweep-into', 'mw-sweep', 'mw-window-open', 'mw-window-close',
		'mw-bin-picking', 'mw-box-close', 'mw-door-lock', 'mw-door-unlock', 'mw-hand-insert',
	],
}
