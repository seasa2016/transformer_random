import logging
import os
import errno
import os

logger = logging.getLogger()

def init_logger(opt,mode="train"):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s %(message)s]")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if(opt.log_file and opt.log_file!=''):
        if(mode=="train"):
            f_name = opt.save_model + opt.log_file
        elif(mode=="test"):
            f_name = opt.log_file
        else:
            raise ValueError('this is an invalid mode')

        if(not os.path.exists(os.path.dirname(f_name))):
            try:
                os.makedirs(os.path.dirname(f_name))
            except OSError as e:
                if(e.errno != errno.EEXIST):
                    raise

        file_handler = logging.FileHandler(f_name)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    if(opt.show):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)

    return logger
