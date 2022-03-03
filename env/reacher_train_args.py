REACHER_TRAIN_ARGS = {

        # general.

    'name'                      : 'reacher',
    'device'                    : 'cuda',
    'logdir'                    : 'runs/',
    'epochs'                    : 5000, #100,
    'max_buffer_size'           : 4096*32,#*3,

        # env.

    'env_steps_per_train'       : 8192,
    'env_max_steps'             : 512,

        # world model.

    'wm_lr'                     : 1e-4,
    'wm_max_grad_norm'          : 10.0,
    'wm_max_grad_skip'          : 20.0,

    'wm_train_samples'          : 4096,
    'wm_minibatch'              : 1024, #128,

    'wm_hid_units'              : 1024,
    'wm_hid_layers'             : 3,
    'wm_window'                 : 4,

    'wm_l1_reg'                 : 0.01,
    'wm_l2_reg'                 : 0.001,

        # policy.

    'po_lr'                     : 1e-4,
    'po_max_grad_norm'          : 10.0,
    'po_max_grad_skip'          : 20.0,

    'po_wm_exploration'         : 0.01, #0.05,
    'po_env_exploration'        : 0.1,

    'po_train_samples'          : 4096,
    'po_minibatch'              : 1024, #256,

    'po_hid_units'              : 1024,
    'po_hid_layers'             : 3,
    'po_window'                 : 32,

    'po_l1_reg'                 : 0.01,
    'po_l2_reg'                 : 0.001,

}