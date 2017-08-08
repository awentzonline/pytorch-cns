def list_of(type_):
    def f(s):
        try:
            return list(map(type_, s.split(',')))
        except:
            raise argparse.ArgumentTypeError('Must be a list of {}'.format(type_))
    return f
