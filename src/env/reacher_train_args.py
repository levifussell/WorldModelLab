REACHER_TRAIN_ARGS = {

        # general.

    'seed'                      : 1234,
    'name'                      : 'reacher',
    'device'                    : 'cuda',
    'logdir'                    : 'runs/',
    'deep_stats'                : False,
    'save_renders'              : True,

        # train.

    'epochs'                    : 1000,
    'max_buffer_size'           : 4096*32,

        # env.

    'env_steps_per_train'       : 8192,
    'env_max_steps'             : 128,

        # world model.

    'wm_lr'                     : 1e-4,
    'wm_max_grad_norm'          : 10.0,
    'wm_max_grad_skip'          : 100.0,

    'wm_train_samples'          : 8192,
    'wm_minibatch'              : 1024,

    'wm_hid_units'              : 512,
    'wm_hid_layers'             : 2,
    'wm_window'                 : 4,

    'wm_l1_reg'                 : 0.01,
    'wm_l2_reg'                 : 0.01,

    'wm_activation'             : 'elu',
    'wm_use_spectral_norm'      : False,

        # policy.

    'po_lr'                     : 1e-4,
    'po_max_grad_norm'          : 10.0,
    'po_max_grad_skip'          : 100.0,

    'po_wm_exploration'         : 0.3, 
    'po_env_exploration'        : 0.3,

    'po_train_samples'          : 8192,
    'po_minibatch'              : 1024,

    'po_hid_units'              : 512,
    'po_hid_layers'             : 2,
    'po_window'                 : 32,

    'po_l1_reg'                 : 1.0,
    'po_l2_reg'                 : 1.0,

}