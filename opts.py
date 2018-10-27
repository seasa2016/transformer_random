import argparse


def model_opts(parser):
    group = parser.add_argument_group('model_embedding')
    group.add_argument('-src_word_vec_size', type=int, default=256,
                       help='Word embedding size for src.')
    group.add_argument('-tar_word_vec_size', type=int, default=256,
                       help='Word embedding size for tgt.')

    group.add_argument('-share_decoder_embeddings', action='store_true',
                       help="""Use a shared weight matrix for the input and
                       output word  embeddings in the decoder.""")
    group.add_argument('-share_embeddings', action='store_true',
                       help="""Share the word embeddings between encoder
                       and decoder. Need to use shared dictionary for this
                       option.""")
    group.add_argument('-position_encoding', action='store_true',
                       help="""Use a sin to mark relative words positions.
                       Necessary for non-RNN style models.
                       """)

    group = parser.add_argument_group('model_embedding_features')
    group.add_argument('-feat_merge', type=str, default='sum',
                       choices=['concat', 'sum', 'mlp'],
                       help="""Merge action for incorporating features embeddings.
                       Options [concat|sum|mlp].""")
    group.add_argument('-feat_vec_size', type=int, default=256,
                       help="""If specified, feature embedding sizes
                       will be set to this. Otherwise, feat_vec_exponent
                       will be used.""")

    # Encoder-Deocder Options
    group = parser.add_argument_group('Model- Encoder-Decoder')
    group.add_argument('-encoder_type', type=str, default='transformer',
                       choices=['rnn', 'brnn', 'mean', 'transformer', 'cnn'],
                       help="""Type of encoder layer to use. Non-RNN layers
                       are experimental. Options are
                       [rnn|brnn|mean|transformer|cnn].""")
    group.add_argument('-decoder_type', type=str, default='transformer',
                       choices=['rnn', 'transformer', 'cnn'],
                       help="""Type of decoder layer to use. Non-RNN layers
                       are experimental. Options are
                       [rnn|transformer|cnn].""")
    group.add_argument('-replace', action='store_true',
                       help="""whether to pass a layer to reduce the dim""")

    group.add_argument('-num_layer', type=int, default=3,
                       help='Number of layers in the encoder')
    group.add_argument('-enc_layer', type=int, default=6,
                       help='Number of layers in the encoder')
    group.add_argument('-dec_layer', type=int, default=3,
                       help='Number of layers in the decoder')
    group.add_argument('-model_dim', type=int, default=256,
                       help='Size of rnn hidden states')
    group.add_argument('-nin_dim_en', type=int, default=1024,
                       help='Size of hidden transformer feed-forward')
    group.add_argument('-nin_dim_de', type=int, default=512,
                       help='Size of hidden transformer feed-forward')
    

    group.add_argument('-dropout', type=float, default=0.1,
                       help="Dropout probability; applied in LSTM stacks.")    

    # Attention options
    group = parser.add_argument_group('Model- Attention')
    group.add_argument('-global_attention', type=str, default='general',
                       choices=['dot', 'general', 'mlp'],
                       help="""The attention type to use:
                       dotprod or general (Luong) or MLP (Bahdanau)""")
    group.add_argument('-self_attn_type', type=str, default="scaled_dot",
                       help="""Self attention type in Transformer decoder
                       layer -- currently "scaled_dot" or "average" """)
    group.add_argument('-num_head', type=int, default=8,
                       help='Number of heads for transformer self-attention')

    group.add_argument('-copy_attn', action='store_true',
                       help='this is for pointer network copy the answer from input')


def train_opts(parser):
    """ Training and saving options """

    group = parser.add_argument_group('General')

    group.add_argument('-save_model', required=True,
                       help="""Model filename (the model will be saved as
                       <save_model>_N.pt where N is the number
                       of steps""")

    group.add_argument('-save_checkpoint_steps', type=int, default=1000,
                       help="""Save a checkpoint every X steps""")
    group.add_argument('-keep_checkpoint', type=int, default=10,
                       help="""Keep X checkpoints (negative: keep all)""")

    # GPU
    group.add_argument('-gpuid', default=[0], nargs='+', type=int,
                       help="Use CUDA on the listed devices.")
    group.add_argument('-gpu_rank', default=0, nargs='+', type=int,
                       help="Rank the current gpu device.")
    group.add_argument('-device_id', default=0, nargs='+', type=int,
                       help="Rank the current gpu device.")
    group.add_argument('-gpu_backend', default='nccl', nargs='+', type=str,
                       help="Type of torch distributed backend")

    group.add_argument('-gpu_verbose_level', default=0, type=int,
                       help="Gives more info on each process per GPU.\n the higher the level, more the information.")

    group.add_argument('-seed', type=int, default=-1,
                       help="""Random seed used for the experiments
                       reproducibility.""")

    # Init options
    group = parser.add_argument_group('Initialization')
    group.add_argument('-param_init', type=float, default=0.1,
                       help="""Parameters are initialized over uniform distribution
                       with support (-param_init, param_init).
                       Use 0 to not use initialization""")
    group.add_argument('-param_init_glorot', action='store_false',
                       help="""Init parameters with xavier_uniform.
                       Required for transfomer.""")

    group.add_argument('-train_from', default=None, type=str,
                       help="""If training from a checkpoint then this is the
                       path to the pretrained model's state_dict.""")

    # Pretrained word vectors
    group.add_argument('-pre_word_vecs_enc',
                       help="""If a valid path is specified, then this will load
                       pretrained word embeddings on the encoder side.
                       See README for specific formatting instructions.""")
    group.add_argument('-pre_word_vecs_dec',
                       help="""If a valid path is specified, then this will load
                       pretrained word embeddings on the decoder side.
                       See README for specific formatting instructions.""")
    # Fixed word vectors
    group.add_argument('-fix_word_vecs_enc',
                       action='store_true',
                       help="Fix word embeddings on the encoder side.")
    group.add_argument('-fix_word_vecs_dec',
                       action='store_true',
                       help="Fix word embeddings on the encoder side.")

    # Optimization options
    group = parser.add_argument_group('Optimization- Type')
    group.add_argument('-batch_size', type=int, default=32,
                       help='Maximum batch size for training')
    group.add_argument('-batch_type', default='tokens',
                       choices=["sents", "tokens"],
                       help="""Batch grouping for batch_size. Standard
                               is sents. Tokens will do dynamic batching""")
    group.add_argument('-normalization', default='tokens',
                       choices=["sents", "tokens"],
                       help='Normalization method of the gradient.')
    #this should be able to fix up the batch size problem
    group.add_argument('-accum_count', type=int, default=2,
                       help="""Accumulate gradient this many times.
                       Approximately equivalent to updating
                       batch_size * accum_count batches at once.
                       Recommended for Transformer.""")

    #training setting
    group = parser.add_argument_group('training_setting')
    group.add_argument('-valid_steps', type=int, default=1000,
                       help='Perfom validation every X steps')
    group.add_argument('-valid_batch_size', type=int, default=32,
                       help='Maximum batch size for validation')
    group.add_argument('-max_generator_batches', type=int, default=16,
                       help="""Maximum batches of words in a sequence to run
                        the generator on in parallel. Higher is faster, but
                        uses more memory.""")
    group.add_argument('-train_steps', type=int, default=300000,
                       help='Number of training steps')
    group.add_argument('-epochs', type=int, default=100,
                       help='Deprecated epochs see train_steps')
    group.add_argument('-optim_method', default='adam',
                       choices=['sgd', 'adagrad', 'adadelta', '',
                                'sparseadam'],
                       help="""Optimization method.""")
    group.add_argument('-max_grad_norm', type=float, default=5,
                       help="""If the norm of the gradient vector exceeds this,
                       renormalize it to have the norm equal to
                       max_grad_norm""")

    group.add_argument('-truncated_decoder', type=int, default=0,
                       help="""Truncated bptt.""")
    group.add_argument('-adam_beta1', type=float, default=0.9,
                       help="""The beta1 parameter used by Adam.
                       Almost without exception a value of 0.9 is used in
                       the literature, seemingly giving good results,
                       so we would discourage changing this value from
                       the default without due consideration.""")
    group.add_argument('-adam_beta2', type=float, default=0.997,
                       help="""The beta2 parameter used by Adam.
                       Typically a value of 0.999 is recommended, as this is
                       the value suggested by the original paper describing
                       Adam, and is also the value adopted in other frameworks
                       such as Tensorflow and Kerras, i.e. see:
                       https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
                       https://keras.io/optimizers/ .
                       Whereas recently the paper "Attention is All You Need"
                       suggested a value of 0.98 for beta2, this parameter may
                       not work well for normal models / default
                       baselines.""")
    group.add_argument('-label_smoothing', type=float, default=0.1,
                       help="""Label smoothing value epsilon.
                       Probabilities of all non-true labels
                       will be smoothed by epsilon / (vocab_size - 1).
                       Set to zero to turn off label smoothing.
                       For more detailed information, see:
                       https://arxiv.org/abs/1512.00567""")
   
    # learning rate
    group = parser.add_argument_group('Optimization- Rate')
    group.add_argument('-learning_rate', type=float, default=0.2,
                       help="""Starting learning rate.
                       Recommended settings: sgd = 1, adagrad = 0.1,
                       adadelta = 1, adam = 0.001""")
    group.add_argument('-learning_rate_decay', type=float, default=1,
                       help="""If update_learning_rate, decay learning rate by
                       this much if (i) perplexity does not decrease on the
                       validation set or (ii) steps have gone past
                       start_decay_steps""")
    group.add_argument('-start_decay_step', type=int, default=16000,
                       help="""Start decaying every decay_steps after
                       start_decay_step""")
    group.add_argument('-decay_steps', type=int, default=5000,
                       help="""Decay every decay_steps""")
    group.add_argument('-decay_method', type=str, default="constant*linear_warmup*rsqrt_decay*rsqrt_hidden_size",
                       choices=['noam'], help="Use a custom decay rate.")
    group.add_argument('-warmup_steps', type=int, default=16000,
                       help="""Number of warmup steps for custom decay.""")

    group = parser.add_argument_group('Logging')
    group.add_argument('-report_every', type=int, default=100,
                       help="Print stats at this interval.")
    group.add_argument('-log_file', type=str, default="/logger",
                       help="Output logs to a file under this path.")
    group.add_argument('-show', action="store_true",
                       help="whether to show on screem.")
    # Use TensorboardX for visualization during training
    group.add_argument('-tensorboard', action="store_false",
                       help="""Use tensorboardX for visualization during training.
                       Must have the library tensorboardX.""")
    group.add_argument("-tensorboard_log_dir", type=str,
                       default="./tensor_log",
                       help="""Log directory for Tensorboard.
                       This is also the name of the run.
                       """)

def translation_opts(parser):
    """ Translation / inference options """
    group = parser.add_argument_group('Model')
    group.add_argument('-test_from',type=str, default='transformer',
                       help='checkpoint for the model.')

    group.add_argument('-model',type=str, default='transformer',
                       help='model for decode.')
    group.add_argument('-pre', action='store_true',
                       help="""use pretrain model""")

    group = parser.add_argument_group('Data')

    group.add_argument('-src', required=True,
                       help="""Source sequence to decode (one line per
                       sequence)""")
    group.add_argument('-tgt', default=None,
                       help='True target sequence in token(optional)')
    group.add_argument('-tgt_truth', default=None,
                       help='True target sequence in word (optional)')
    group.add_argument('-output', default='pred.txt',
                       help="""Path to output the predictions (each line will
                       be the decoded sequence""")
    group.add_argument('-report_bleu', action='store_true',
                       help="""Report bleu score after translation,
                       call tools/multi-bleu.perl on command line""")
    group.add_argument('-report_rouge', action='store_true',
                       help="""Report rouge 1/2/3/L/SU4 score after translation
                       call tools/test_rouge.py on command line""")

    # Options most relevant to summarization.
    group.add_argument('-dynamic_dict', action='store_true',
                       help="Create dynamic dictionaries")
    group.add_argument('-share_vocab', action='store_true',
                       help="Share source and target vocabulary")

    group = parser.add_argument_group('Beam')
    group.add_argument('-fast', action="store_true",
                       help="""Use fast beam search (some features may not be
                       supported!)""")
    group.add_argument('-beam_size', type=int, default=3,
                       help='Beam size')
    group.add_argument('-min_length', type=int, default=5,
                       help='Minimum prediction length')
    group.add_argument('-max_length', type=int, default=100,
                       help='Maximum prediction length.')
    group.add_argument('-max_sent_length', action=DeprecateAction,
                       help="Deprecated, use `-max_length` instead")

    # Alpha and Beta values for Google Length + Coverage penalty
    # Described here: https://arxiv.org/pdf/1609.08144.pdf, Section 7
    group.add_argument('-stepwise_penalty', action='store_true',
                       help="""Apply penalty at every decoding step.
                       Helpful for summary penalty.""")
    group.add_argument('-length_penalty', default='none',
                       choices=['none', 'wu', 'avg'],
                       help="""Length Penalty to use.""")
    group.add_argument('-coverage_penalty', default='none',
                       choices=['none', 'wu', 'summary'],
                       help="""Coverage Penalty to use.""")
    group.add_argument('-alpha', type=float, default=0.,
                       help="""Google NMT length penalty parameter
                        (higher = longer generation)""")
    group.add_argument('-beta', type=float, default=-0.,
                       help="""Coverage penalty parameter""")
    group.add_argument('-block_ngram_repeat', type=int, default=1,
                       help='Block repetition of ngrams during decoding.')
    group.add_argument('-ignore_when_blocking', nargs='+', type=str,
                       default=[],
                       help="""Ignore these strings when blocking repeats.
                       You want to block sentence delimiters.""")
    group.add_argument('-replace_unk', action="store_true",
                       help="""Replace the generated UNK tokens with the
                       source token that had highest attention weight. If
                       phrase_table is provided, it will lookup the
                       identified source token and give the corresponding
                       target token. If it is not provided(or the identified
                       source token does not exist in the table) then it
                       will copy the source token""")

    group = parser.add_argument_group('Logging')
    group.add_argument('-verbose', action="store_true",
                       help='Print scores and predictions for each sentence')
    group.add_argument('-show', action="store_true",
                       help="whether to show on screem.")
    group.add_argument('-log_file', type=str, default="./logger",
                       help="Output logs to a file under this path.")
    group.add_argument('-attn_debug', action="store_true",
                       help='Print best attn for each word')
    group.add_argument('-dump_beam', type=str, default="",
                       help='File to dump beam information to.')
    group.add_argument('-n_best', type=int, default=1,
                       help="""If verbose is set, will output the n_best
                       decoded sentences""")

    group = parser.add_argument_group('Efficiency')
    group.add_argument('-batch_size', type=int, default=16,
                       help='Batch size')
    group.add_argument('-gpu', type=int, default=0,
                       help="Device to run on")




def add_md_help_argument(parser):
    """ md help parser """
    parser.add_argument('-md', action=MarkdownHelpAction,
                        help='print Markdown-formatted help text and exit.')

class MarkdownFormmatter(argparse.HelpFormatter):
    def _format_usage(self,usage,actions,groups,prefix):
        return ""

    def format_help(self):
        print(self._prog)
        self._root_section.heading = "#Options: {0}".format(self._prog)

        return super(MarkdownFormmatter,self).format_help()
    
    def start_section(self,heading):
        super(MarkdownFormmatter,self).start_section(
            "### **{0}**".format(heading)
        )

    def _format_action(self,action):
        if(action.dest == "help" or action.dest == "md"):
            return ""
        line = []
        line.append("* **-{0} {1}** ".format(action.dest,str(action.default) if action.default else "[]"))
    
        if(action.help):
            help_text = self._expand_help(action)
            line.extend(self._split_lines(help_text,80))
        line.extend(["",""])
        return '\n'.join(line)


class MarkdownHelpAction(argparse.Action):

    def __init__(self,option_strings,dest=argparse.SUPPRESS,
                    default=argparse.SUPPRESS,**kwargs):
        super(MarkdownHelpAction,self).__init__(
            option_strings = option_strings,
            dest=dest,
            default=default,
            nargs=0,
            **kwargs
        )
    def __call__(self,parser,namespace,values,option_string=None):
        parser.formatter_class = MarkdownFormmatter
        parser.print_help()
        parser.exit()
    

class DeprecateAction(argparse.Action):
    """ Deprecate action """

    def __init__(self, option_strings, dest, help=None, **kwargs):
        super(DeprecateAction, self).__init__(option_strings, dest, nargs=0,
                                              help=help, **kwargs)

    def __call__(self, parser, namespace, values, flag_name):
        help = self.help if self.mdhelp is not None else ""
        msg = "Flag '%s' is deprecated. %s" % (flag_name, help)
        raise argparse.ArgumentTypeError(msg)
