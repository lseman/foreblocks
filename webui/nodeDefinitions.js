const NODE_TYPES = {
  // ===== DATA NODES =====
  data_input: {
    name: 'Data Input',
    category: 'Data',
    color: 'bg-gradient-to-br from-blue-600 to-blue-700',
    outputs: ['X_train', 'Y_train'],
    config: { 
      seq_len: 50, 
      target_len: 10,
      total_len: 1000,
      data_type: 'synthetic_sine'
    },
    py: {
      imports: [''],
      // ctor: 'None',
      var_prefix: 'datainput',
      bind: {
        kwargs: {
          seq_len: '@config:seq_len',
          target_len: '@config:target_len',
          total_len: '@config:total_len',
          data_type: '@config:data_type'
        }
      }
    }
  },

  output: {
    name: 'Output',
    category: 'Visualization',
    color: 'bg-gradient-to-br from-green-600 to-green-700',
    inputs: ['trained_model', 'X_val', 'Y_val'],
    optional_inputs: ['X_val', 'Y_val'],
    config: { 
      plot: true,
      metrics: true
    },
    py: {
      imports: [''],
      // ctor: 'None',
      var_prefix: 'outputnode',
      bind: {
        kwargs: {
          plot: '@config:plot',
          metrics: '@config:metrics',
          trained_model: '@input:trained_model',
          X_val: '@input:X_val',
          Y_val: '@input:Y_val'
        }
      }
    }
  },

  trainer: {
    name: 'Trainer',
    category: 'Training',
    color: 'bg-gradient-to-br from-amber-600 to-orange-700',
    inputs: ['model', 'X_train', 'Y_train'],
    optional_inputs: ['X_train', 'Y_train'],
    outputs: ['trained_model'],
    config: {
      epochs: 5,
      batch_size: 64,
      learning_rate: 0.001,
      weight_decay: 0.0,
      use_amp: true
    },
    py: {
      imports: ['from foreblocks.core.training.trainer import Trainer'],
      ctor: 'Trainer',
      var_prefix: 'trainer',
      role: 'trainer',
      bind: {
        kwargs: {
          model: '@input:model'
        }
      },
      output_map: {
        trained_model: '@self'
      }
    }
  },
};
