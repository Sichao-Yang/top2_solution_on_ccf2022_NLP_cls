import logging

def get_logger(filename, 
               verbosity='info', 
               logname=None, 
               method='w', 
               stream=True,
               ):
    level_dict = {'debug': logging.DEBUG, 'info': logging.INFO, 'warn': logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(logname)

    if stream:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    # rotateHandler = RFHandler(filename, method, fsize, last_k)
    # rotateHandler.setFormatter(formatter)
    # logger.addHandler(rotateHandler)
    
    fh = logging.FileHandler(filename, method)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    logger.setLevel(level_dict[verbosity])

    return logger