CARTPOLE_BALANCE_TRAIN_ARGS = {

        # general.

    'seed'                      : 1234,
    'name'                      : 'cartpole_balance',
    'device'                    : 'cpu',
    'logdir'                    : 'runs/',
    'deep_stats'                : False,

        # train.

    'epochs'                    : 1000, #100,
    'max_buffer_size'           : 4096*32,#*3,

        # env.

    'env_steps_per_train'       : 8192,
    'env_max_steps'             : 1000,

        # world model.

    'wm_lr'                     : 1e-4,
    'wm_max_grad_norm'          : 10.0,
    'wm_max_grad_skip'          : 100.0,

    'wm_train_samples'          : 8192,
    'wm_minibatch'              : 1024, #128,

    'wm_hid_units'              : 512, #1024,
    'wm_hid_layers'             : 2,
    'wm_window'                 : 8,

    'wm_l1_reg'                 : 0.0001, #0.01,
    'wm_l2_reg'                 : 0.0001, #0.001,

        # policy.

    'po_lr'                     : 1e-4,
    'po_max_grad_norm'          : 10.0,
    'po_max_grad_skip'          : 100.0,

    'po_wm_exploration'         : 1.0, #0.05,
    'po_env_exploration'        : 1.0,

    'po_train_samples'          : 8192,
    'po_minibatch'              : 1024, #256,

    'po_hid_units'              : 512, #1024,
    'po_hid_layers'             : 2,
    'po_window'                 : 128,

    'po_l1_reg'                 : 0.1,#0.01,
    'po_l2_reg'                 : 0.1,#0.001,

}