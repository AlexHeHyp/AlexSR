def create_model(opt):
    model = opt['model']

    if model == 'sr':
        from .SR_model import SRModel
        m = SRModel(opt)
        print('Model [%s] is created.' % m.__class__.__name__)
        return m
    else:
        raise NotImplementedError('Model [%s] not recognized.' % model)


