import argparse

from model.util.Logger import logger,init_logger
from model.translate.translator import build_translator

import opts

def main(opt):
    translator = build_translator(opt, report_score=True,logger=logger)
    translator.translate(src_path=opt.src,
                         tgt_path=opt.tgt,
                         attn_debug=opt.attn_debug)


if(__name__ == "__main__"):
    parser = argparse.ArgumentParser(
        description='test.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    opts.add_md_help_argument(parser)
    opts.model_opts(parser)
    opts.translation_opts(parser)

    opt = parser.parse_args()
    logger = init_logger(opt.log_file)
    main(opt)
